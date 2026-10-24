"""PneumoniaMNIST seed43/44: one shared implementation, staged execution.

Models, run_epoch, augmentation and entropy match the inspected Colab definitions.
Importing this file does not load datasets, train, infer or create directories.
"""
from pathlib import Path
import copy
import hashlib
import json
import platform
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TF
import medmnist
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt


# These definitions reproduce the inspected existing notebook, once for all runs.
class SmallCNN(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c2, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


class DropoutCNN(nn.Module):
    def __init__(self, in_channels, channels, num_classes, dropout_p=0.3):
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = torch.flatten(self.pool(self.features(x)), 1)
        return self.classifier(self.dropout(x))


def fixed_blur(image, sigma):
    if sigma == 0:
        return image
    return TF.gaussian_blur(image, kernel_size=[9, 9],
                            sigma=[float(sigma), float(sigma)])


class GrayDataset(Dataset):
    def __init__(self, source, training_blur=False):
        self.images = np.asarray(source.imgs)
        self.labels = np.asarray(source.labels).reshape(-1)
        self.training_blur = training_blur

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0) / 255.0
        if self.training_blur and random.random() < 0.5:
            image = fixed_blur(image, random.uniform(0.3, 1.0))
        return image, torch.tensor(int(self.labels[index]), dtype=torch.long)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Existing run_epoch logic; globals device/positive_class_id made explicit."""
    training = optimizer is not None
    model.train() if training else model.eval()
    loss_sum = correct = count = 0
    ys, ps = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            assert x.dtype == torch.float32 and x.shape[1:] == (1, 28, 28)
            assert y.ndim == 1 and y.dtype == torch.long
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('非有限loss：停止并保留失败运行。')
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach().item() * len(y)
            correct += (logits.detach().argmax(1) == y).sum().item()
            count += len(y)
            if not training:
                ys.append(y.cpu().numpy())
                ps.append(logits.softmax(1)[:, 1].cpu().numpy())
    assert count == len(loader.dataset)
    result = dict(loss=loss_sum / count, accuracy=correct / count)
    if not training:
        result['AUROC'] = float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))
    return result


def entropy(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -q * np.log(q) - (1 - q) * np.log(1 - q)


def divide(a, b):
    return float(a / b) if b else np.nan


def metrics(y, p):
    p = np.asarray(p)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(AUROC=float(roc_auc_score(y, p)), AP=float(average_precision_score(y, p)),
                accuracy=float(np.mean(pred == y)), sensitivity=divide(tp, tp + fn),
                specificity=divide(tn, tn + fp), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def unique_dir(parent, name):
    path, suffix = Path(parent) / name, 1
    while path.exists():
        path = Path(parent) / f'{name}_{suffix:03d}'
        suffix += 1
    path.mkdir(exist_ok=False)
    return path


class RepeatStudy:
    SIGMAS = [0.0, 0.5, 1.0, 1.5]
    PCTS = [0, 10, 20, 30, 50]
    MC_SEEDS = {43: 1043, 44: 1044}
    T = 30
    MI_TOL = 1e-10
    ATOL, RTOL = 1e-6, 1e-5

    def __init__(self, project):
        self.project = Path(project)
        assert self.project.is_dir()
        self.runs = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Check actual files before reading exact confirmed filenames.
        self.inventory = []
        for name in ['baseline_seed42', 'val_blur_pilot_seed42', 'blur_aug_seed42',
                     'dropout_seed42', 'mc_dropout_val_seed42', 'confidence_review_sim_seed42']:
            folder = self.project / name
            assert folder.is_dir(), f'缺少历史目录：{folder}'
            self.inventory.extend(dict(directory=name, file=p.name) for p in sorted(folder.iterdir()) if p.is_file())
        print(pd.DataFrame(self.inventory).to_string(index=False))
        self.base = json.loads((self.project / 'baseline_seed42/experiment_config.json').read_text())
        self.drop = json.loads((self.project / 'dropout_seed42/config.json').read_text())
        self.blur_config = json.loads((self.project / 'blur_aug_seed42/config.json').read_text())
        self.mc_config = json.loads((self.project / 'mc_dropout_val_seed42/config.json').read_text())
        self.labels = {int(k): v for k, v in medmnist.INFO['pneumoniamnist']['label'].items()}
        assert self.labels == {0: 'normal', 1: 'pneumonia'}
        self.pre = copy.deepcopy(self.base['preprocessing'])
        assert self.pre == dict(input_height=28, input_width=28, channels=1, input_layout='NCHW',
                                source_dtype='uint8', tensor_dtype='float32',
                                normalization='x / 255.0', resize=None, augmentation=None)
        assert self.base['model_config'] == dict(in_channels=1, channels=[16, 32], num_classes=2)
        assert self.drop['model_config'] == dict(in_channels=1, channels=[16, 32], num_classes=2, dropout_p=.3)
        assert self.mc_config['MC']['T'] == self.T
        assert self.mc_config['entropy']['MI_negative_tolerance'] == self.MI_TOL
        assert self.mc_config['sigmas'] == self.SIGMAS
        assert self.mc_config['blur']['kernel_size'] == [9, 9]
        self.versions = dict(python=platform.python_version(), torch=str(torch.__version__),
                             torchvision=str(torchvision.__version__), medmnist=medmnist.__version__,
                             numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                             matplotlib=matplotlib.__version__, device=str(self.device), cuda=torch.version.cuda)
        # Official train/val only. Do not instantiate the test split.
        self.train = medmnist.PneumoniaMNIST(split='train', size=28, download=True, transform=None)
        self.val = medmnist.PneumoniaMNIST(split='val', size=28, download=True, transform=None)
        assert self.train.imgs.shape == (4708, 28, 28) and self.val.imgs.shape == (524, 28, 28)
        assert self.train.imgs.dtype == self.val.imgs.dtype == np.uint8
        self.y = np.asarray(self.val.labels).reshape(-1).astype(np.int64)
        self.x = torch.from_numpy(self.val.imgs.copy()).unsqueeze(1).float() / 255.0
        self.val_loader = DataLoader(GrayDataset(self.val), batch_size=64, shuffle=False, num_workers=0)
        self.criterion = nn.CrossEntropyLoss()
        order = np.random.default_rng(42).permutation(524)
        self.tie_rank = np.empty(524, dtype=int)
        self.tie_rank[order] = np.arange(524)
        old_ties = pd.read_csv(self.project / 'mc_dropout_val_seed42/tie_break_ranks.csv').sort_values('val_index')
        assert np.array_equal(old_ties.tie_rank, self.tie_rank)
        self.old_hashes = {str(p): sha(p) for row in self.inventory
                           for p in [self.project / row['directory'] / row['file']]}
        self.campaign = unique_dir(self.project, 'repeats_seed43_44')
        write_json(self.campaign / 'historical_hashes.json', self.old_hashes)
        write_json(self.campaign / 'fixed_plan.json', dict(training_seeds=[43, 44],
                   models=['baseline', 'blur_aug', 'dropout'], optimizer='Adam', lr=.001,
                   batch_size=64, epochs=10, num_workers=0, checkpoint_selection='minimum clean val loss',
                   augmentation=dict(probability=.5, kernel=[9, 9], sigma_uniform=[.3, 1.0]),
                   dropout=dict(position='after flatten before classifier', p=.3),
                   validation_sigmas=self.SIGMAS, threshold=.5, T=self.T, MC_seeds=self.MC_SEEDS,
                   historical_MC_seed=42, tie_seed=42, review_percentages=self.PCTS,
                   review_rounding='(percent*524)//100', random_reference='理论随机参照：原始总错误数/524',
                   versions=self.versions, metric_tolerance=dict(atol=self.ATOL, rtol=self.RTOL)))
        (self.campaign / 'shared_implementation.py').write_text(Path(__file__).read_text(), encoding='utf-8')
        print('新批次目录：', self.campaign, '设备：', self.device)
        x, y = next(iter(self.val_loader))
        print('输入：', x.shape, x.dtype, '标签：', y.shape, y.dtype)
        for kind in ['baseline', 'dropout']:
            model = self.make_model(kind)
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert n == 4866
            print(kind, '实际参数：', n)

    def make_model(self, kind, config=None):
        config = copy.deepcopy(config or (self.drop if kind == 'dropout' else self.base)['model_config'])
        cls = DropoutCNN if kind == 'dropout' else SmallCNN
        return cls(**config).to(self.device)

    def _failed(self, directory, stage, error):
        path = Path(directory) / f'failure_{stage}.json'
        write_json(path, dict(stage=stage, exception=repr(error), traceback=traceback.format_exc(),
                             retained=True, explanation='原始失败运行保留；不自动调参或按表现重跑。'))

    def train_one(self, kind, seed):
        assert kind in ['baseline', 'blur_aug', 'dropout'] and seed in [43, 44]
        if (kind, seed) != ('baseline', 43):
            assert (self.campaign / 'pilot_seed43_baseline_passed.json').is_file(), '先完成seed43 baseline恢复核对。'
        directory = unique_dir(self.project, f'{kind}_seed{seed}')
        self.runs[(kind, seed)] = directory
        write_json(self.campaign / f'register_{directory.name}.json', dict(model=kind, seed=seed, path=str(directory)))
        started = time.perf_counter()
        try:
            config = copy.deepcopy((self.drop if kind == 'dropout' else self.base)['model_config'])
            cfg = dict(model=kind, seed=seed, model_class='DropoutCNN' if kind == 'dropout' else 'SmallCNN',
                       model_config=config, preprocessing=self.pre, label_map=self.labels,
                       optimizer='Adam', learning_rate=.001, batch_size=64, epochs=10, num_workers=0,
                       checkpoint_selection='minimum clean validation loss', from_scratch=True,
                       train_shuffle_generator_seed=seed, validation_shuffle=False,
                       seed_reset='before initialization and again immediately before training',
                       augmentation=dict(probability=.5, sigma_uniform=[.3, 1.0], kernel=[9, 9]) if kind == 'blur_aug' else None,
                       dropout=dict(position='after flatten before classifier', p=.3) if kind == 'dropout' else None,
                       MC_inference_seed=self.MC_SEEDS[seed] if kind == 'dropout' else None,
                       versions=self.versions, campaign=str(self.campaign),
                       source_sha256=sha(self.campaign / 'shared_implementation.py'),
                       validation_image_sha256=hashlib.sha256(self.val.imgs.tobytes()).hexdigest(),
                       validation_label_sha256=hashlib.sha256(self.y.tobytes()).hexdigest())
            write_json(directory / 'config.json', cfg)
            seed_all(seed)
            model = self.make_model(kind, config)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4866
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(GrayDataset(self.train, training_blur=(kind == 'blur_aug')),
                                batch_size=64, shuffle=True, generator=generator, num_workers=0)
            seed_all(seed)
            generator.manual_seed(seed)
            history, best_loss = [], float('inf')
            train_start = time.perf_counter()
            for epoch in range(1, 11):
                tr = run_epoch(model, loader, self.criterion, self.device, optimizer)
                va = run_epoch(model, self.val_loader, self.criterion, self.device)
                row = dict(epoch=epoch, train_loss=tr['loss'], train_accuracy=tr['accuracy'],
                           val_loss=va['loss'], val_accuracy=va['accuracy'], val_AUROC=va['AUROC'])
                history.append(row)
                pd.DataFrame([row]).to_csv(directory / 'epoch_results.csv', index=False,
                                          mode='x' if epoch == 1 else 'a', header=(epoch == 1))
                improved = va['loss'] < best_loss
                if improved:
                    best_loss = va['loss']
                    torch.save(dict(epoch=epoch, model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                    model_class=cfg['model_class'], model_config=config, label_map=self.labels,
                                    preprocessing=self.pre, training=cfg, metrics_at_best_epoch=row,
                                    trainable_parameters=4866), directory / 'best_model.pt')
                print(f'{kind} seed{seed} epoch{epoch:02d}: val loss={va["loss"]:.6f}, acc={va["accuracy"]:.6f}, AUROC={va["AUROC"]:.6f}' + (' 保存最佳' if improved else ''), flush=True)
            training_seconds = time.perf_counter() - train_start
            hist = pd.read_csv(directory / 'epoch_results.csv')
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, metric in zip(axes[:2], ['loss', 'accuracy']):
                for split in ['train', 'val']:
                    ax.plot(hist.epoch, hist[f'{split}_{metric}'], label=split)
                ax.set(xlabel='Epoch', ylabel=metric); ax.legend(); ax.grid(alpha=.2)
            axes[2].plot(hist.epoch, hist.val_AUROC); axes[2].set(xlabel='Epoch', ylabel='Clean validation AUROC', ylim=(0, 1))
            fig.tight_layout(); fig.savefig(directory / 'training_curves.png', dpi=180); plt.close(fig)
            restore_start = time.perf_counter()
            _, ck, actual = self.restore(directory)
            restore_seconds = time.perf_counter() - restore_start
            timing = dict(model=kind, seed=seed, training_seconds=training_seconds,
                          restore_seconds=restore_seconds, total_seconds=time.perf_counter() - started,
                          best_epoch=int(ck['epoch']), clean_validation=actual, passed=True)
            write_json(directory / 'training_restore_completed.json', timing)
            if (kind, seed) == ('baseline', 43):
                write_json(self.campaign / 'pilot_seed43_baseline_passed.json', dict(path=str(directory), **timing))
            print('训练与恢复核对通过：', timing, '目录：', directory, flush=True)
            return directory
        except Exception as error:
            self._failed(directory, 'training_or_restore', error)
            raise

    def restore(self, directory):
        directory = Path(directory)
        ck = torch.load(directory / 'best_model.pt', map_location=self.device, weights_only=True)
        cfg = json.loads((directory / 'config.json').read_text())
        assert ck['model_config'] == cfg['model_config']
        assert ck['preprocessing'] == self.pre
        assert {int(k): v for k, v in ck['label_map'].items()} == self.labels
        model = self.make_model(cfg['model'], ck['model_config'])
        model.load_state_dict(ck['model_state_dict'], strict=True)
        if cfg['model'] == 'dropout':
            assert isinstance(model.dropout, nn.Dropout) and model.dropout.p == .3
        actual = run_epoch(model, self.val_loader, self.criterion, self.device)
        history = pd.read_csv(directory / 'epoch_results.csv')
        assert int(history.loc[history.val_loss.idxmin(), 'epoch']) == ck['epoch']
        saved_row = history.loc[history.epoch == ck['epoch']].iloc[0]
        checks = []
        for metric in ['loss', 'accuracy', 'AUROC']:
            for source, target in [('checkpoint', ck['metrics_at_best_epoch']['val_' + metric]),
                                   ('epoch_csv', saved_row['val_' + metric])]:
                checks.append(dict(metric=metric, source=source, actual=actual[metric], saved=float(target),
                                   difference=actual[metric] - target,
                                   passed=bool(np.isclose(actual[metric], target, atol=self.ATOL, rtol=self.RTOL))))
        checks = pd.DataFrame(checks)
        if not (directory / 'restore_check.csv').exists():
            checks.to_csv(directory / 'restore_check.csv', index=False, mode='x')
        assert checks.passed.all(), '恢复指标不一致：停止评价，保留原指标。'
        model.eval()
        return model, ck, actual

    def _deterministic(self, model, x):
        model.eval(); probabilities = []; loss_sum = 0.
        with torch.no_grad():
            for start in range(0, 524, 64):
                logits = model(x[start:start + 64].to(self.device))
                target = torch.as_tensor(self.y[start:start + 64], device=self.device, dtype=torch.long)
                loss_sum += nn.functional.cross_entropy(logits, target).item() * len(target)
                probabilities.append(logits.softmax(1)[:, 1].cpu().numpy())
        return np.concatenate(probabilities).astype(np.float64), loss_sum / 524

    def review(self, y, pred, score, model, seed, sigma, strategy, prediction):
        order = np.lexsort((self.tie_rank, score))
        errors = (pred != y).astype(int); original_errors = int(errors.sum())
        retained_n = np.arange(1, 525)
        kept_errors = np.cumsum(errors[order[::-1]])
        common = dict(model=model, seed=seed, sigma=sigma, strategy=strategy, prediction=prediction)
        curve = pd.DataFrame(dict(**common, retained_n=retained_n, reviewed_n=524-retained_n,
                                  coverage=retained_n/524, retained_errors=kept_errors,
                                  retained_risk=kept_errors/retained_n,
                                  theoretical_random_risk=original_errors/524))
        rows = []
        for pct in self.PCTS:
            m = (pct*524)//100; kept = order[m:]; referred = order[:m]
            tn, fp, fn, tp = confusion_matrix(y[kept], pred[kept], labels=[0, 1]).ravel()
            rows.append(dict(**common, target_review_pct=pct, retained_n=len(kept), reviewed_n=m,
                             coverage=len(kept)/524, retained_risk=float((fp+fn)/len(kept)),
                             TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                             sensitivity=divide(tp, tp+fn), specificity=divide(tn, tn+fp),
                             original_errors=original_errors, removed_errors=int(original_errors-fp-fn),
                             removed_error_fraction=divide(original_errors-fp-fn, original_errors),
                             referred_pneumonia_fraction=divide(int(y[referred].sum()), m),
                             theoretical_random_risk=original_errors/524, random_reference_type='理论随机参照'))
        ranks = np.empty(524, dtype=int); ranks[order] = np.arange(1, 525)
        ranking = pd.DataFrame(dict(**common, val_index=np.arange(524), true_label=y,
                                    predicted_label=pred, review_rank=ranks, tie_rank=self.tie_rank))
        return pd.DataFrame(rows), curve, ranking

    def evaluate_one(self, directory):
        directory = Path(directory)
        assert (directory / 'training_restore_completed.json').is_file()
        out = unique_dir(directory, 'validation_evaluation')
        try:
            started = time.perf_counter()
            model, ck, restored = self.restore(directory)
            cfg = json.loads((directory / 'config.json').read_text())
            kind, seed = cfg['model'], cfg['seed']
            source_hash = sha(directory / 'best_model.pt')
            weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            frames, class_rows, review_frames, curves, rankings, modes, mi_checks = [], [], [], [], [], [], []
            for sigma in self.SIGMAS:
                if kind == 'dropout':
                    seed_all(self.MC_SEEDS[seed])  # Once per condition, never per forward.
                x = self.x if sigma == 0 else torch.stack([fixed_blur(im, sigma) for im in self.x])
                assert x.shape == self.x.shape and x.dtype == torch.float32
                det_p, loss = self._deterministic(model, x)
                det_metrics = metrics(self.y, det_p)
                if sigma == 0:
                    for key, value in dict(loss=loss, accuracy=det_metrics['accuracy'], AUROC=det_metrics['AUROC']).items():
                        assert np.isclose(value, restored[key], atol=self.ATOL, rtol=self.RTOL), '原图恢复核对不通过，停止。'
                common = dict(model=kind, seed=seed, sigma=sigma)
                class_rows.append(dict(**common, prediction='deterministic', loss=loss, **det_metrics))
                frame = pd.DataFrame(dict(val_index=np.arange(524), true_label=self.y, **common,
                                          deterministic_p=det_p, deterministic_pred=(det_p>=.5).astype(int),
                                          deterministic_confidence=np.maximum(det_p, 1-det_p)))
                strategies = [('A', 'deterministic', frame.deterministic_pred.to_numpy(), frame.deterministic_confidence.to_numpy())]
                if kind == 'dropout':
                    model.eval()
                    for module in model.modules():
                        if isinstance(module, nn.Dropout): module.train()
                    for name, module in model.named_modules():
                        assert module.training == isinstance(module, nn.Dropout)
                        modes.append(dict(**common, module=name or '<root>', training=module.training, type=type(module).__name__))
                    draws = np.empty((524, self.T), dtype=np.float64)
                    with torch.no_grad():
                        for start in range(0, 524, 64):
                            batch = x[start:start+64].to(self.device)
                            draws[start:start+len(batch)] = torch.stack([model(batch).softmax(1)[:, 1] for _ in range(self.T)], 1).cpu().numpy()
                    assert np.isfinite(draws).all() and ((draws>=0)&(draws<=1)).all()
                    mean_p = draws.mean(1)
                    predictive = entropy(mean_p); expected = entropy(draws).mean(1); raw_mi = predictive-expected
                    mi_checks.append(dict(sigma=sigma, min_raw_MI=float(raw_mi.min()),
                                          negative_count=int((raw_mi<0).sum()), significant_negative_count=int((raw_mi < -self.MI_TOL).sum())))
                    assert not (raw_mi < -self.MI_TOL).any(), '显著负MI，停止；不能直接截断。'
                    frame['mean_p'] = mean_p; frame['mc_pred'] = (mean_p>=.5).astype(int)
                    frame['mc_confidence'] = np.maximum(mean_p, 1-mean_p)
                    frame['predictive_entropy'] = predictive; frame['expected_entropy'] = expected
                    frame['MI_raw'] = raw_mi; frame['MI'] = np.maximum(raw_mi, 0.)
                    frame['probability_variance'] = draws.var(1, ddof=0)
                    frame = pd.concat([frame, pd.DataFrame(draws, columns=[f'p_t{i+1:02d}' for i in range(self.T)])], axis=1)
                    class_rows.append(dict(**common, prediction='mc_mean', loss=np.nan, **metrics(self.y, mean_p)))
                    # B/C use the exact same array, only their ranking score changes.
                    mc_pred = frame.mc_pred.to_numpy()
                    strategies.extend([('B', 'mc_mean', mc_pred, frame.mc_confidence.to_numpy()),
                                       ('C', 'mc_mean', mc_pred, -frame.MI.to_numpy())])
                frames.append(frame)
                for strategy, pred_type, pred, score in strategies:
                    review, curve, ranking = self.review(self.y, pred, score, kind, seed, sigma, strategy, pred_type)
                    review_frames.append(review); curves.append(curve); rankings.append(ranking)
                print(kind, seed, 'sigma', sigma, '评价完成', flush=True)
            model.eval()
            assert all(torch.equal(weights[k], v.detach().cpu()) for k, v in model.state_dict().items())
            assert sha(directory / 'best_model.pt') == source_hash
            data = dict(predictions=pd.concat(frames, ignore_index=True), classification=pd.DataFrame(class_rows),
                        review=pd.concat(review_frames, ignore_index=True), curves=pd.concat(curves, ignore_index=True),
                        rankings=pd.concat(rankings, ignore_index=True))
            assert len(data['predictions']) == 2096
            twenties = data['review'].query('target_review_pct==20')
            assert (twenties.retained_n==420).all() and (twenties.reviewed_n==104).all()
            for key, table in data.items(): table.to_csv(out / f'{key}.csv', index=False, mode='x')
            pd.DataFrame(modes).to_csv(out / 'module_modes.csv', index=False, mode='x')
            pd.DataFrame(mi_checks).to_csv(out / 'MI_checks.csv', index=False, mode='x')
            write_json(out / 'config.json', dict(training_directory=str(directory), model=kind, training_seed=seed,
                       checkpoint_sha256=source_hash, sigmas=self.SIGMAS, threshold=.5, batch_size=64,
                       T=self.T if kind=='dropout' else None, MC_inference_seed=self.MC_SEEDS[seed] if kind=='dropout' else None,
                       MC_seed_reset='once per sigma; no reseeding per forward; batch then T forwards',
                       entropy_clip=[1e-7,1-1e-7], entropy_log='natural', MI_negative_tolerance=self.MI_TOL,
                       tie_seed=42, tie_rule='same label-independent permutation and ascending tie_rank as seed42',
                       random_reference='理论随机参照：each prediction set original_total_errors/524; not 1000-trial mean',
                       versions=self.versions, curve_excludes_coverage_zero=True))
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for ax, sigma in zip(axes.flat, self.SIGMAS):
                for strategy in sorted(data['curves'].strategy.unique()):
                    curve = data['curves'].query('sigma==@sigma and strategy==@strategy')
                    ax.plot(curve.coverage, curve.retained_risk, label=strategy)
                    ax.axhline(curve.theoretical_random_risk.iloc[0], linestyle='--', alpha=.35,
                               label=f'{strategy} theoretical random')
                ax.set(xlim=(1,0), ylim=(0,1), xlabel='Coverage 1 -> 0 (more review ->)',
                       ylabel='Retained risk [0,1]', title=f'{kind} seed{seed}, sigma={sigma:g}')
                ax.legend(fontsize=7); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(out / 'risk_coverage.png', dpi=180); plt.close(fig)
            self.verify_evaluation(out)
            write_json(out / 'completed.json', dict(passed=True, seconds=time.perf_counter()-started,
                       checkpoint_unchanged=True, weights_unchanged=True,
                       classification_rows=len(data['classification']), curve_rows=len(data['curves'])))
            write_json(directory / f'evaluation_{out.name}_completed.json', dict(path=str(out), passed=True))
            print('保存并重新读取核验通过：', out, flush=True)
            return out
        except Exception as error:
            self._failed(out, 'evaluation', error)
            raise

    def verify_evaluation(self, out):
        out = Path(out)
        ps = pd.read_csv(out / 'predictions.csv'); cl = pd.read_csv(out / 'classification.csv')
        rv = pd.read_csv(out / 'review.csv'); curves = pd.read_csv(out / 'curves.csv')
        ranks = pd.read_csv(out / 'rankings.csv')
        assert not ps.isna().any().any()
        checks = []
        for sigma in self.SIGMAS:
            group = ps.query('sigma==@sigma').sort_values('val_index')
            assert len(group)==524 and np.array_equal(group.val_index, np.arange(524)) and np.array_equal(group.true_label, self.y)
            pairs = [('deterministic','deterministic_p','deterministic_pred')]
            if 'mean_p' in group:
                draws = group[[f'p_t{i+1:02d}' for i in range(self.T)]].to_numpy()
                assert np.allclose(draws.mean(1), group.mean_p, atol=1e-12, rtol=1e-12)
                assert np.allclose(entropy(group.mean_p)-entropy(draws).mean(1), group.MI_raw, atol=1e-12, rtol=1e-12)
                assert np.allclose(np.maximum(group.MI_raw,0), group.MI, atol=1e-12, rtol=1e-12)
                pairs.append(('mc_mean','mean_p','mc_pred'))
                b=ranks.query('sigma==@sigma and strategy=="B"').sort_values('val_index')
                c=ranks.query('sigma==@sigma and strategy=="C"').sort_values('val_index')
                assert np.array_equal(b.predicted_label,c.predicted_label) and np.array_equal(b.predicted_label,group.mc_pred)
            for pred_type, pcol, lcol in pairs:
                assert np.array_equal((group[pcol]>=.5).astype(int), group[lcol])
                saved = cl.query('sigma==@sigma and prediction==@pred_type').iloc[0]
                for key,value in metrics(self.y,group[pcol].to_numpy()).items():
                    passed=bool(np.isclose(value,saved[key],atol=self.ATOL,rtol=self.RTOL,equal_nan=True))
                    checks.append(dict(sigma=sigma,prediction=pred_type,metric=key,passed=passed))
            for strategy in ranks.strategy.unique():
                ranking = ranks.query('sigma==@sigma and strategy==@strategy').sort_values('val_index')
                pred = ranking.predicted_label.to_numpy(); errors=(pred!=self.y).astype(int)
                order=np.argsort(ranking.review_rank.to_numpy())
                curve=curves.query('sigma==@sigma and strategy==@strategy').sort_values('retained_n')
                n=np.arange(1,525)
                assert np.array_equal(curve.retained_n,n)
                assert np.allclose(curve.retained_risk,np.cumsum(errors[order[::-1]])/n,atol=1e-12)
                for pct in self.PCTS:
                    row=rv.query('sigma==@sigma and strategy==@strategy and target_review_pct==@pct').iloc[0]
                    m=(pct*524)//100; kept=order[m:]
                    tn,fp,fn,tp=confusion_matrix(self.y[kept],pred[kept],labels=[0,1]).ravel()
                    assert [row.TN,row.FP,row.FN,row.TP]==[tn,fp,fn,tp]
                    assert np.isclose(row.retained_risk,(fp+fn)/len(kept),atol=1e-12)
                    assert np.isclose(row.theoretical_random_risk,errors.mean(),atol=1e-12)
        check_table=pd.DataFrame(checks); assert check_table.passed.all()
        check_table.to_csv(out/'saved_metric_check.csv',index=False,mode='x')
        return True

    def historical_unchanged(self):
        assert all(Path(path).is_file() and sha(path)==digest for path,digest in self.old_hashes.items())
        return len(self.old_hashes)


def summarize(study, evaluation_directories):
    """Read real saved results only. Refuse incomplete or ambiguous three-seed sets."""
    evaluations = [Path(p) for p in evaluation_directories]
    assert len(evaluations) == 6 and len(set(evaluations)) == 6
    required = {(kind, seed) for kind in ['baseline','blur_aug','dropout'] for seed in [43,44]}
    found, classification, reviews, inputs = set(), [], [], []
    failure_rows = []
    for out in evaluations:
        done = json.loads((out/'completed.json').read_text()); assert done['passed']
        config = json.loads((out/'config.json').read_text())
        key=(config['model'],config['training_seed'])
        assert key not in found and key in required; found.add(key)
        assert config['sigmas']==study.SIGMAS and config['threshold']==.5
        assert config['tie_seed']==42
        if key[0]=='dropout':
            assert config['T']==30 and config['MC_inference_seed']==study.MC_SEEDS[key[1]]
        for filename in ['config.json','predictions.csv','classification.csv','review.csv','rankings.csv','curves.csv']:
            path=out/filename
            inputs.append(dict(path=str(path),sha256=sha(path)))
        classification.append(pd.read_csv(out/'classification.csv'))
        reviews.append(pd.read_csv(out/'review.csv').query('target_review_pct==20').copy())
        for p in out.parent.glob('failure_*.json'):
            failure_rows.append(dict(path=str(p),details=p.read_text()))
        for p in out.glob('failure_*.json'):
            failure_rows.append(dict(path=str(p),details=p.read_text()))
    assert found==required, '缺少运行，不能计算三个seed汇总。'
    old_review=pd.read_csv(study.project/'confidence_review_sim_seed42/review_summary_40rows.csv')
    old_mc_review=pd.read_csv(study.project/'mc_dropout_val_seed42/review_summary_60rows.csv')
    for kind,folder,summary_name in [('baseline','val_blur_pilot_seed42','summary.csv'),
                                     ('blur_aug','blur_aug_seed42','blur_summary.csv'),
                                     ('dropout','mc_dropout_val_seed42','classification_summary_8rows.csv')]:
        path=study.project/folder
        original=pd.read_csv(path/summary_name)
        if 'prediction' not in original: original['prediction']='deterministic'
        original['model']=kind; original['seed']=42
        classification.append(original)
        pred_table=pd.read_csv(path/'per_sample_predictions.csv')
        for sigma in study.SIGMAS:
            group=pred_table.query('sigma==@sigma').sort_values('val_index')
            assert len(group)==524 and np.array_equal(group.val_index,np.arange(524))
            assert np.array_equal(group.true_label,study.y)
            assert not group.isna().any().any()
            pairs=[('deterministic','deterministic_p','deterministic_pred'),('mc_mean','mean_p','mc_pred')] if kind=='dropout' else [('deterministic','pneumonia_probability','predicted_label')]
            for prediction,pcol,lcol in pairs:
                pred=(group[pcol].to_numpy()>=.5).astype(int)
                assert np.array_equal(pred,group[lcol])
                saved=original.query('sigma==@sigma and prediction==@prediction').iloc[0]
                for metric,value in metrics(study.y,group[pcol].to_numpy()).items():
                    assert np.isclose(value,saved[metric],atol=study.ATOL,rtol=study.RTOL,equal_nan=True)
        if kind=='dropout':
            rv=old_mc_review.query('target_review_pct==20').copy()
            rv['model']='dropout'; rv['seed']=42
            old_ranks=pd.read_csv(path/'sample_review_rankings.csv')
            for sigma in study.SIGMAS:
                b=old_ranks.query('sigma==@sigma and strategy=="B"').sort_values('val_index')
                c=old_ranks.query('sigma==@sigma and strategy=="C"').sort_values('val_index')
                assert np.array_equal(b.predicted_label,c.predicted_label)
        else:
            rv=old_review.query('model==@kind and requested_review_percent==20').copy()
            rv=rv.rename(columns={'auto_count':'retained_n','review_count':'reviewed_n',
                                  'requested_review_percent':'target_review_pct',
                                  'original_total_errors':'original_errors'})
            rv['seed']=42; rv['strategy']='A'; rv['prediction']='deterministic'
        # Convert to exactly labelled theoretical expectation, never the saved simulation mean.
        rv['theoretical_random_risk']=rv.original_errors/524
        rv['random_reference_type']='理论随机参照'
        rv=rv.drop(columns=[c for c in rv if 'random' in c and c not in ['theoretical_random_risk','random_reference_type']])
        reviews.append(rv)
        for filename in ['config.json',summary_name,'per_sample_predictions.csv']:
            p=path/filename;inputs.append(dict(path=str(p),sha256=sha(p)))
    cl=pd.concat(classification,ignore_index=True)
    rv=pd.concat(reviews,ignore_index=True)
    assert len(cl)==48 and len(rv)==60
    assert (rv.retained_n==420).all() and (rv.reviewed_n==104).all()
    assert np.allclose(rv.retained_risk,(rv.FP+rv.FN)/420,atol=1e-12)
    assert np.allclose(rv.coverage,420/524,atol=1e-12)
    assert np.allclose(rv.theoretical_random_risk,rv.original_errors/524,atol=1e-12)
    assert not cl.duplicated(['model','seed','sigma','prediction']).any()
    assert not rv.duplicated(['model','seed','sigma','strategy']).any()
    selected=['model','seed','sigma','strategy','prediction','coverage','retained_n','reviewed_n',
              'retained_risk','TN','FP','FN','TP','removed_error_fraction','theoretical_random_risk']
    rv=rv[selected].copy()
    main=rv.merge(cl[['model','seed','sigma','prediction','AUROC','AP','sensitivity','specificity']],
                  on=['model','seed','sigma','prediction'],how='left',validate='many_to_one')
    assert not main[['AUROC','AP','sensitivity','specificity','retained_risk']].isna().any().any()
    main['MC_inference_seed']=main.apply(lambda row: (42 if row.seed==42 else study.MC_SEEDS[row.seed])
                                        if row.model=='dropout' and row.prediction=='mc_mean' else np.nan,axis=1)
    main=main.sort_values(['model','strategy','sigma','seed']).reset_index(drop=True)
    # pandas std uses the sample standard deviation, ddof=1.
    summary=main.groupby(['model','strategy','prediction','sigma'])[['AUROC','AP','sensitivity','specificity','retained_risk']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    counts=main.groupby(['model','strategy','prediction','sigma']).seed.nunique()
    assert (counts==3).all()
    summary['n_seeds']=3
    mc=rv.query('model=="dropout"').pivot(index=['seed','sigma'],columns='strategy',values='retained_risk')
    delta=(mc.C-mc.B).rename('C_minus_B_retained_risk').reset_index()
    delta['C_minus_B_percentage_points']=delta.C_minus_B_retained_risk*100
    delta_summary=delta.groupby('sigma').agg(
        n_seeds=('seed','nunique'),mean_difference=('C_minus_B_retained_risk','mean'),
        sample_std_difference=('C_minus_B_retained_risk','std'),
        mean_percentage_points=('C_minus_B_percentage_points','mean'),
        sample_std_percentage_points=('C_minus_B_percentage_points','std')).reset_index()
    assert (delta_summary.n_seeds==3).all()
    out=unique_dir(study.project,'summary_seed42_43_44')
    for name,table in [('classification_per_seed.csv',cl),('review20_per_seed.csv',rv),
                       ('main_metrics_per_seed.csv',main),('mean_sample_std.csv',summary),
                       ('dropout_C_minus_B_per_seed.csv',delta),('dropout_C_minus_B_mean_sample_std.csv',delta_summary),
                       ('input_manifest.csv',pd.DataFrame(inputs))]:
        table.to_csv(out/name,index=False,mode='x')
    # Include failed attempts from this campaign, including ones not in completed selections.
    for registration in study.campaign.glob('register_*.json'):
        directory=Path(json.loads(registration.read_text())['path'])
        for p in directory.rglob('failure_*.json'):
            if str(p) not in {r['path'] for r in failure_rows}:
                failure_rows.append(dict(path=str(p),details=p.read_text()))
    pd.DataFrame(failure_rows,columns=['path','details']).to_csv(out/'failure_records.csv',index=False,mode='x')
    preserved=study.historical_unchanged()
    limitations=('三个seed使用相同、已用于探索的验证集。样本标准差ddof=1只描述这些训练和MC推理随机设置下的波动，'
                 '不是人群泛化置信区间，不作显著性或临床安全性声明。MC seed42=42、seed43=1043、seed44=1044；'
                 '不能把训练与MC随机性的影响分别归因。全部方案事先固定，不删除差运行。'
                 '理论随机参照为各套预测原始总错误率，不是1000次模拟均值。'
                 '不假设人工一定正确，不计算模型+人工系统最终准确率。')
    (out/'explanation_zh.md').write_text(limitations,encoding='utf-8')
    write_json(out/'verification.json',dict(passed=True,classification_rows=len(cl),review20_rows=len(rv),
               main_rows=len(main),summary_rows=len(summary),delta_rows=len(delta),n_seeds=3,ddof=1,
               historical_files_unchanged=preserved,failure_records=len(failure_rows),versions=study.versions,
               campaign=str(study.campaign),evaluation_directories=[str(p) for p in evaluations]))
    # Verify saved summaries against saved per-seed rows, not just in-memory objects.
    reread=pd.read_csv(out/'main_metrics_per_seed.csv')
    saved=pd.read_csv(out/'mean_sample_std.csv')
    for _,row in saved.iterrows():
        group=reread[(reread.model==row.model)&(reread.strategy==row.strategy)&(reread.sigma==row.sigma)]
        assert set(group.seed)=={42,43,44}
        for metric in ['AUROC','AP','sensitivity','specificity','retained_risk']:
            assert np.isclose(group[metric].mean(),row[metric+'_mean'],atol=1e-12)
            assert np.isclose(group[metric].std(ddof=1),row[metric+'_std'],atol=1e-12)
    print('三个seed真实文件汇总及重读核对通过：',out)
    print(limitations)
    return out,main,summary,delta,delta_summary
