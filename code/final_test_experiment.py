"""Frozen final evaluation. Import performs no data loading or inference.
Three methods copied from the verified repeats implementation with only N and
index-name generalization; the original source remains unmodified.
"""
from repeat_experiments import *
from datetime import datetime, timezone
import inspect

SIGMAS=[0.,.5,1.,1.5]
PCTS=[0,10,20,30,50]

def utc(): return datetime.now(timezone.utc).isoformat()
def load_json(path): return json.loads(Path(path).read_text())
def csv(path,frame): frame.to_csv(path,index=False,mode='x')
def close(a,b): return np.isclose(a,b,atol=1e-6,rtol=1e-5,equal_nan=True)

class FinalTest:
    SIGMAS=SIGMAS
    PCTS=PCTS
    T=30
    ATOL=1e-6
    RTOL=1e-5
    MI_TOL=1e-10

    def __init__(self,project):
        self.project=Path(project)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pre={"input_height":28,"input_width":28,"channels":1,"input_layout":"NCHW","source_dtype":"uint8","tensor_dtype":"float32","normalization":"x / 255.0","resize":None,"augmentation":None}
        self.labels={int(k):v for k,v in medmnist.INFO['pneumoniamnist']['label'].items()}
        assert self.labels=={0:'normal',1:'pneumonia'}
        self.records=[]; self.protected={}; self.configs={}; self.mc_seeds={}
        self.versions=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,medmnist=medmnist.__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,device=str(self.device))

    def preflight(self):
        for seed in (42,43,44):
            for kind in ('baseline','blur_aug','dropout'):
                directory=self.project/f'{kind}_seed{seed}'
                names=sorted(p.name for p in directory.iterdir())
                assert 'best_model.pt' in names and 'epoch_results.csv' in names
                cf=directory/('experiment_config.json' if (kind,seed)==('baseline',42) else 'config.json')
                assert cf.name in names
                cfg=load_json(cf); cp=directory/'best_model.pt'
                ck=torch.load(cp,map_location=self.device,weights_only=True)
                expected=dict(in_channels=1,channels=[16,32],num_classes=2)
                if kind=='dropout': expected['dropout_p']=.3
                assert cfg['model_config']==ck['model_config']==expected
                assert cfg['preprocessing']==ck['preprocessing']==self.pre
                assert {int(k):v for k,v in cfg['label_map'].items()}==ck['label_map']==self.labels
                training=cfg.get('training',cfg)
                assert training['optimizer']=='Adam' and training['learning_rate']==.001
                assert training['batch_size']==64 and training['epochs']==10 and training['num_workers']==0
                assert training.get('seed',cfg.get('seed'))==seed
                selection=training.get('selection_rule',training.get('checkpoint_selection'))
                assert selection in ('minimum validation loss','minimum clean validation loss')
                if kind=='blur_aug':
                    if seed==42:
                        a=cfg['training_augmentation']
                        assert a['blur_probability']==a['keep_original_probability']==.5 and a['kernel_size']==[9,9] and a['sigma_distribution']=='Uniform(0.3,1.0)' and a['other_augmentation'] is None
                    else: assert cfg['augmentation']==dict(probability=.5,sigma_uniform=[.3,1.],kernel=[9,9])
                if kind=='dropout':
                    assert cfg['dropout']['p']==.3 and 'flatten' in cfg['dropout']['position'] and ('linear' in cfg['dropout']['position'] or 'classifier' in cfg['dropout']['position'])
                history=pd.read_csv(directory/'epoch_results.csv')
                loss_col='clean_val_loss' if (kind,seed)==('blur_aug',42) else 'val_loss'
                assert len(history)==10 and history.epoch.is_unique
                assert int(history.loc[history[loss_col].idxmin(),'epoch'])==int(ck['epoch'])
                assert close(history[loss_col].min(),ck['metrics_at_best_epoch'][loss_col])
                model=self.load_model(kind,ck)
                assert sum(p.numel() for p in model.parameters())==4866
                if kind=='dropout':
                    mcpath=self.project/'mc_dropout_val_seed42/config.json' if seed==42 else directory/'validation_evaluation/config.json'
                    m=load_json(mcpath)
                    if seed==42:
                        assert m['MC']['T']==30 and m['MC']['batch_size']==64
                        mcseed=m['MC']['seed']
                        assert m['sigmas']==SIGMAS and m['blur']['kernel_size']==[9,9]
                        assert m['entropy']['clip']==[1e-7,1-1e-7] and m['entropy']['MI_negative_tolerance']==1e-10
                    else:
                        assert m['T']==30 and m['sigmas']==SIGMAS and m['tie_seed']==42 and m['threshold']==.5
                        mcseed=m['MC_inference_seed']; assert mcseed==cfg['MC_inference_seed']
                    self.mc_seeds[seed]=mcseed; self.protected[str(mcpath)]=sha(mcpath)
                for p in (cp,cf,directory/'epoch_results.csv'): self.protected[str(p)]=sha(p)
                self.records.append(dict(model=kind,seed=seed,path=str(cp),sha256=sha(cp),config_path=str(cf),config_sha256=sha(cf),best_epoch=int(ck['epoch']),parameters=4866,selection=selection))
                self.configs[f'{kind}_seed{seed}']=cfg
        assert len(self.records)==9
        assert self.mc_seeds=={42:42,43:1043,44:1044}
        return pd.DataFrame(self.records)

    def load_model(self,kind,ck):
        cls=DropoutCNN if kind=='dropout' else SmallCNN
        model=cls(**ck['model_config']).to(self.device)
        model.load_state_dict(ck['model_state_dict'],strict=True)
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        return model

    def tie(self,n):
        self.n=n
        order=np.random.default_rng(42).permutation(n)
        self.tie_rank=np.empty(n,dtype=int);self.tie_rank[order]=np.arange(n)

    def validate_generalization(self):
        # Only saved validation files. No datasets or test inference.
        out=self.project/'dropout_seed43/validation_evaluation'
        ps=pd.read_csv(out/'predictions.csv'); old=pd.read_csv(out/'review.csv'); oldcv=pd.read_csv(out/'curves.csv')
        self.tie(524); checks=[]
        for sigma in SIGMAS:
            g=ps.query('sigma==@sigma').sort_values('val_index')
            y=g.true_label.to_numpy(); assert np.array_equal(g.val_index,np.arange(524))
            draws=g[[f'p_t{i+1:02d}' for i in range(30)]].to_numpy()
            assert np.allclose(draws.mean(1),g.mean_p,atol=1e-12,rtol=1e-12)
            assert np.allclose(entropy(g.mean_p)-entropy(draws).mean(1),g.MI_raw,atol=1e-12,rtol=1e-12)
            for strategy,pred,score in [('A',g.deterministic_pred,np.maximum(g.deterministic_p,1-g.deterministic_p)),('B',g.mc_pred,np.maximum(g.mean_p,1-g.mean_p)),('C',g.mc_pred,-g.MI)]:
                rv,cv,rk=self.review(y,pred.to_numpy(),np.asarray(score),'dropout',43,sigma,strategy,'deterministic' if strategy=='A' else 'mc_mean')
                a=old.query('sigma==@sigma and strategy==@strategy').reset_index(drop=True)
                b=oldcv.query('sigma==@sigma and strategy==@strategy').reset_index(drop=True)
                for col in rv.select_dtypes(include='number').columns:
                    assert np.allclose(rv[col],a[col],atol=1e-12,rtol=1e-12,equal_nan=True),col
                for col in cv.select_dtypes(include='number').columns:
                    assert np.allclose(cv[col],b[col],atol=1e-12,rtol=1e-12,equal_nan=True),col
                checks.append(dict(sigma=sigma,strategy=strategy,passed=True))
        # N-generalization sanity checks use synthetic labels, never reported as experiment results.
        for n in (7,101,625):
            self.tie(n);y=np.arange(n)%2;pred=(np.arange(n)%3==0).astype(int)
            rv,cv,_=self.review(y,pred,np.arange(n,dtype=float),'synthetic',0,0,'A','deterministic')
            assert len(cv)==n and cv.coverage.min()>0 and cv.coverage.max()==1
            assert np.array_equal(rv.reviewed_n,[(p*n)//100 for p in PCTS])
            assert np.allclose(rv.retained_risk,(rv.FP+rv.FN)/rv.retained_n)
        return pd.DataFrame(checks)

    def freeze(self):
        assert len(self.records)==9
        checks=self.validate_generalization()
        self.out=unique_dir(self.project,'final_test_seed42_43_44')
        source=Path(__file__); shared=Path(inspect.getfile(SmallCNN))
        for p,name in ((source,'final_test_implementation.py'),(shared,'verified_shared_implementation.py')):
            with (self.out/name).open('x',encoding='utf-8') as f:f.write(p.read_text())
        self.source_hashes={str(source):sha(source),str(shared):sha(shared)}
        self.protocol=dict(frozen_at_utc=utc(),status='frozen before final test dataset loading and inference',registration='Settings determined after exploratory validation; NOT preregistration at study start.',dataset=dict(name='PneumoniaMNIST',split='official test; train/val retain official split',preprocessing=self.pre,label_map=self.labels,sample_count='read from official test after freeze; never copy validation N'),checkpoints=self.records,selection='minimum clean validation loss',models=dict(baseline='SmallCNN',blur_aug='same SmallCNN; historical training 50% unchanged, 50% 9x9 blur Uniform(0.3,1.0)',dropout='DropoutCNN; only flatten -> Dropout(p=0.3) -> linear; no added parameters'),sigmas=SIGMAS,blur=dict(function='torchvision.transforms.functional.gaussian_blur',kernel_size=[9,9],sigma_zero='skip entirely',input='CPU CHW float32 /255 in [0,1]',padding='default reflection',resize=False,per_image_minmax=False),MC=dict(T=30,seeds_by_training_seed=self.mc_seeds,seed_reset='once per sigma before deterministic/MC, never per forward',loop_order='batch64 then T forwards',aggregation='softmax every pass, average probabilities in float64',mode='model.eval(); only nn.Dropout.train(); no_grad, no updates'),entropy=dict(log='natural',clip=[1e-7,1-1e-7],dtype='float64',MI_negative_tolerance=1e-10,negative_MI='stop if below tolerance; only roundoff negatives clipped'),threshold=.5,primary='Dropout, identical MC mean labels: target20% retained risk C(high MI) minus B(low confidence), paired per seed then mean/sample SD',secondary='classification metrics, all other review percentages, full risk-coverage curves',review=dict(percentages=PCTS,rounding='floor(N*percent/100) using actual test N',tie='one default_rng(42).permutation(N), unrelated to labels, ascending tie rank, shared all models/conditions',confidence='max(p,1-p)',risk='retained errors/retained N; omit coverage=0',zero_denominator='NaN',random_reference='THEORETICAL RANDOM REFERENCE: corresponding prediction original total errors/N, not simulation'),statistics=dict(seeds=[42,43,44],ddof=1,pooling=False,interpretation='seed SD describes stochastic-setting variation on same test samples; not population generalization CI; no significance claims'),test_policy='no training, hyperparameter/model/threshold adjustment, or result-driven rerun; record implementation error and pause',versions=self.versions,implementation_sha256=self.source_hashes,protected_inputs_sha256=self.protected,scope='controlled artificial blur; no clinical safety claim; no assumption human reviews are correct; no final human+model system accuracy')
        write_json(self.out/'final_protocol.json',self.protocol)
        text=['# 冻结方案与最终测试评价','这些设置是在查看探索性验证结果后确定的，不能称为研究开始前预注册。','主要比较：同一Dropout模型、同一MC平均预测标签，20%复核时C高MI减B低置信度的retained risk。','仅使用9个既有最低干净验证loss检查点；不训练、不改设置、不因结果不好重跑。','数据采用官方测试划分，28×28灰度float32除以255；人数在冻结后实际读取。','高斯核9×9，sigma=0（完全跳过）、0.5、1.0、1.5。Dropout在flatten后linear前，p=0.3，T=30。','MC种子沿用'+str(self.mc_seeds)+'，每条件开始重设一次，逐次softmax再平均；只有Dropout开启。','分类阈值p>=0.5。复核比例0/10/20/30/50%，人数floor(N×比例)。平局default_rng(42)一次无标签排列。','随机参照为对应预测原始总错误率的理论期望。所有零分母写NaN，coverage=0不计算risk。','逐seed报告后计算均值和样本标准差ddof=1；C-B先逐seed配对相减。禁止合并seed样本当独立患者；SD不是泛化置信区间。','次要分析包括分类指标、其他复核比例及完整risk–coverage曲线。','人工复核只模拟排序，不假定人工全对。人工模糊不代表真实医院数据变化，不声明统计显著性或临床安全。','## 9个检查点及SHA256']
        text += [f"- {r['path']} | SHA256 {r['sha256']} | epoch {r['best_epoch']}" for r in self.records]
        text += ['## 完整机器可读方案','```json',json.dumps(self.protocol,ensure_ascii=False,indent=2),'```']
        with (self.out/'final_protocol.md').open('x',encoding='utf-8') as f:f.write(chr(10).join(text))
        csv(self.out/'checkpoint_manifest.csv',pd.DataFrame(self.records));csv(self.out/'pretest_validation_logic_regression.csv',checks)
        write_json(self.out/'freeze_receipt.json',dict(at_utc=utc(),protocol_sha256=sha(self.out/'final_protocol.json'),markdown_sha256=sha(self.out/'final_protocol.md'),test_loaded=False,validation_regression_passed=True))
        print('冻结已保存；尚未读取测试集：',self.out,flush=True)
        return self.out

    def _deterministic(self, model, x):
        model.eval(); probabilities = []; loss_sum = 0.
        with torch.no_grad():
            for start in range(0, self.n, 64):
                logits = model(x[start:start + 64].to(self.device))
                target = torch.as_tensor(self.y[start:start + 64], device=self.device, dtype=torch.long)
                loss_sum += nn.functional.cross_entropy(logits, target).item() * len(target)
                probabilities.append(logits.softmax(1)[:, 1].cpu().numpy())
        return np.concatenate(probabilities).astype(np.float64), loss_sum / self.n

    def review(self, y, pred, score, model, seed, sigma, strategy, prediction):
        order = np.lexsort((self.tie_rank, score))
        errors = (pred != y).astype(int); original_errors = int(errors.sum())
        retained_n = np.arange(1, (self.n+1))
        kept_errors = np.cumsum(errors[order[::-1]])
        common = dict(model=model, seed=seed, sigma=sigma, strategy=strategy, prediction=prediction)
        curve = pd.DataFrame(dict(**common, retained_n=retained_n, reviewed_n=self.n-retained_n,
                                  coverage=retained_n/self.n, retained_errors=kept_errors,
                                  retained_risk=kept_errors/retained_n,
                                  theoretical_random_risk=original_errors/self.n))
        rows = []
        for pct in self.PCTS:
            m = (pct*self.n)//100; kept = order[m:]; referred = order[:m]
            tn, fp, fn, tp = confusion_matrix(y[kept], pred[kept], labels=[0, 1]).ravel()
            rows.append(dict(**common, target_review_pct=pct, retained_n=len(kept), reviewed_n=m,
                             coverage=len(kept)/self.n, retained_risk=float((fp+fn)/len(kept)),
                             TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                             sensitivity=divide(tp, tp+fn), specificity=divide(tn, tn+fp),
                             original_errors=original_errors, removed_errors=int(original_errors-fp-fn),
                             removed_error_fraction=divide(original_errors-fp-fn, original_errors),
                             referred_pneumonia_fraction=divide(int(y[referred].sum()), m),
                             theoretical_random_risk=original_errors/self.n, random_reference_type='理论随机参照'))
        ranks = np.empty(self.n, dtype=int); ranks[order] = np.arange(1, (self.n+1))
        ranking = pd.DataFrame(dict(**common, test_index=np.arange(self.n), true_label=y,
                                    predicted_label=pred, review_rank=ranks, tie_rank=self.tie_rank))
        return pd.DataFrame(rows), curve, ranking

    def verify_evaluation(self, out):
        out = Path(out)
        ps = pd.read_csv(out / 'predictions.csv'); cl = pd.read_csv(out / 'classification.csv')
        rv = pd.read_csv(out / 'review.csv'); curves = pd.read_csv(out / 'curves.csv')
        ranks = pd.read_csv(out / 'rankings.csv')
        assert not ps.isna().any().any()
        checks = []
        for sigma in self.SIGMAS:
            group = ps.query('sigma==@sigma').sort_values('test_index')
            assert len(group)==self.n and np.array_equal(group.test_index, np.arange(self.n)) and np.array_equal(group.true_label, self.y)
            pairs = [('deterministic','deterministic_p','deterministic_pred')]
            if 'mean_p' in group:
                draws = group[[f'p_t{i+1:02d}' for i in range(self.T)]].to_numpy()
                assert np.allclose(draws.mean(1), group.mean_p, atol=1e-12, rtol=1e-12)
                assert np.allclose(entropy(group.mean_p)-entropy(draws).mean(1), group.MI_raw, atol=1e-12, rtol=1e-12)
                assert np.allclose(np.maximum(group.MI_raw,0), group.MI, atol=1e-12, rtol=1e-12)
                pairs.append(('mc_mean','mean_p','mc_pred'))
                b=ranks.query('sigma==@sigma and strategy=="B"').sort_values('test_index')
                c=ranks.query('sigma==@sigma and strategy=="C"').sort_values('test_index')
                assert np.array_equal(b.predicted_label,c.predicted_label) and np.array_equal(b.predicted_label,group.mc_pred)
            for pred_type, pcol, lcol in pairs:
                assert np.array_equal((group[pcol]>=.5).astype(int), group[lcol])
                saved = cl.query('sigma==@sigma and prediction==@pred_type').iloc[0]
                for key,value in metrics(self.y,group[pcol].to_numpy()).items():
                    passed=bool(np.isclose(value,saved[key],atol=self.ATOL,rtol=self.RTOL,equal_nan=True))
                    checks.append(dict(sigma=sigma,prediction=pred_type,metric=key,passed=passed))
            for strategy in ranks.strategy.unique():
                ranking = ranks.query('sigma==@sigma and strategy==@strategy').sort_values('test_index')
                pred = ranking.predicted_label.to_numpy(); errors=(pred!=self.y).astype(int)
                order=np.argsort(ranking.review_rank.to_numpy())
                curve=curves.query('sigma==@sigma and strategy==@strategy').sort_values('retained_n')
                n=np.arange(1,(self.n+1))
                assert np.array_equal(curve.retained_n,n)
                assert np.allclose(curve.retained_risk,np.cumsum(errors[order[::-1]])/n,atol=1e-12)
                for pct in self.PCTS:
                    row=rv.query('sigma==@sigma and strategy==@strategy and target_review_pct==@pct').iloc[0]
                    m=(pct*self.n)//100; kept=order[m:]
                    tn,fp,fn,tp=confusion_matrix(self.y[kept],pred[kept],labels=[0,1]).ravel()
                    assert [row.TN,row.FP,row.FN,row.TP]==[tn,fp,fn,tp]
                    assert np.isclose(row.retained_risk,(fp+fn)/len(kept),atol=1e-12)
                    assert np.isclose(row.theoretical_random_risk,errors.mean(),atol=1e-12)
        check_table=pd.DataFrame(checks); assert check_table.passed.all()
        check_table.to_csv(out/'saved_metric_check.csv',index=False,mode='x')
        return True

    def integrity(self):
        receipt=load_json(self.out/'freeze_receipt.json')
        assert sha(self.out/'final_protocol.json')==receipt['protocol_sha256']
        assert sha(self.out/'final_protocol.md')==receipt['markdown_sha256']
        assert all(sha(p)==h for p,h in self.source_hashes.items())
        assert all(sha(p)==h for p,h in self.protected.items())

    def run(self):
        self.integrity()
        # Exclusive start marker: an existing run can never silently rerun inference.
        write_json(self.out/'test_started.json',dict(at_utc=utc(),protocol_sha256=sha(self.out/'final_protocol.json')))
        try:
            info=medmnist.INFO['pneumoniamnist']
            source=getattr(medmnist,info['python_class'])(split='test',size=28,download=True,transform=None,target_transform=None)
            images=np.asarray(source.imgs);self.y=np.asarray(source.labels).reshape(-1).astype(int)
            self.tie(len(self.y));assert self.n==len(source)==info['n_samples']['test']
            assert images.shape==(self.n,28,28) and images.dtype==np.uint8
            assert set(np.unique(self.y))==set(self.labels)
            self.x=torch.as_tensor(images,dtype=torch.float32).unsqueeze(1)/255.0
            assert self.x.shape==(self.n,1,28,28) and 0<=self.x.min()<=self.x.max()<=1
            data=dict(at_utc=utc(),N=self.n,shape=list(images.shape),dtype=str(images.dtype),labels={str(k):int((self.y==k).sum()) for k in self.labels},image_sha256=hashlib.sha256(images.tobytes()).hexdigest(),label_sha256=hashlib.sha256(self.y.tobytes()).hexdigest())
            write_json(self.out/'actual_test_data.json',data)
            counts=pd.DataFrame([dict(target_review_pct=p,reviewed_n=(p*self.n)//100,retained_n=self.n-(p*self.n)//100,coverage=(self.n-(p*self.n)//100)/self.n) for p in PCTS])
            csv(self.out/'actual_review_counts.csv',counts)
            csv(self.out/'test_index_labels_tie_ranks.csv',pd.DataFrame(dict(test_index=np.arange(self.n),true_label=self.y,tie_rank=self.tie_rank)))
            print('官方测试集实际人数与标签：',data,flush=True); print(counts.to_string(index=False),flush=True)
            self.outputs=[]
            for r in self.records:self.outputs.append(self.evaluate_record(r))
            self.aggregate()
            self.integrity()
            write_json(self.out/'verification.json',dict(passed=True,at_utc=utc(),N=self.n,checkpoints_unchanged=9,protected_inputs_unchanged=len(self.protected),classification_rows=48,review_rows=300,review20_rows=60,curve_rows=60*self.n,paired_delta_rows=12,protocol_unchanged=True,seed_count=3,ddof=1,training_performed=False))
            print('最终测试完成且重读核对通过：',self.out,flush=True)
        except Exception as error:
            write_json(self.out/'failure_record.json',dict(at_utc=utc(),type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),action='PAUSED. No automatic correction, retraining or inference rerun. Preserve all results.'))
            raise
        return self.out

    def evaluate_record(self,r):
        began=time.perf_counter();kind=r['model'];seed=r['seed'];cp=Path(r['path'])
        assert sha(cp)==r['sha256'];ck=torch.load(cp,map_location=self.device,weights_only=True)
        model=self.load_model(kind,ck);weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        out=self.out/f'{kind}_seed{seed}';out.mkdir(exist_ok=False)
        frames=[];class_rows=[];reviews=[];curves=[];rankings=[];modes=[];mi_checks=[]
        write_json(out/'config.json',dict(model=kind,training_seed=seed,checkpoint=r,protocol_sha256=sha(self.out/'final_protocol.json'),MC_seed=self.mc_seeds[seed] if kind=='dropout' else None,N=self.n,versions=self.versions))
        for sigma in SIGMAS:
            if kind=='dropout':seed_all(self.mc_seeds[seed])
            x=self.x if sigma==0 else torch.stack([fixed_blur(im,sigma) for im in self.x])
            assert x.shape==self.x.shape and x.dtype==torch.float32 and torch.isfinite(x).all()
            det_p,loss=self._deterministic(model,x)
            common=dict(model=kind,seed=seed,sigma=sigma)
            class_rows.append(dict(**common,prediction='deterministic',loss=loss,**metrics(self.y,det_p)))
            frame=pd.DataFrame(dict(**common,test_index=np.arange(self.n),true_label=self.y,deterministic_p=det_p,deterministic_pred=(det_p>=.5).astype(int),deterministic_confidence=np.maximum(det_p,1-det_p)))
            strategies=[('A','deterministic',frame.deterministic_pred.to_numpy(),frame.deterministic_confidence.to_numpy())]
            if kind=='dropout':
                model.eval()
                for module in model.modules():
                    if isinstance(module,nn.Dropout):module.train()
                for name,module in model.named_modules():
                    assert module.training==isinstance(module,nn.Dropout)
                    modes.append(dict(**common,module=name or '<root>',type=type(module).__name__,training=module.training))
                draws=np.empty((self.n,30),dtype=np.float64)
                with torch.no_grad():
                    for start in range(0,self.n,64):
                        batch=x[start:start+64].to(self.device)
                        draws[start:start+len(batch)]=torch.stack([model(batch).softmax(1)[:,1] for _ in range(30)],1).cpu().numpy()
                assert np.isfinite(draws).all() and ((draws>=0)&(draws<=1)).all()
                mean_p=draws.mean(1);predictive=entropy(mean_p);expected=entropy(draws).mean(1);raw_mi=predictive-expected
                mi_checks.append(dict(**common,min_raw_MI=float(raw_mi.min()),negative_count=int((raw_mi<0).sum()),significant_negative_count=int((raw_mi < -self.MI_TOL).sum())))
                assert not (raw_mi < -self.MI_TOL).any(),'显著负MI：停止。'
                frame['mean_p']=mean_p;frame['mc_pred']=(mean_p>=.5).astype(int);frame['mc_confidence']=np.maximum(mean_p,1-mean_p)
                frame['predictive_entropy']=predictive;frame['expected_entropy']=expected;frame['MI_raw']=raw_mi;frame['MI']=np.maximum(raw_mi,0.)
                frame['probability_variance']=draws.var(1,ddof=0)
                frame=pd.concat([frame,pd.DataFrame(draws,columns=[f'p_t{i+1:02d}' for i in range(30)])],axis=1)
                class_rows.append(dict(**common,prediction='mc_mean',loss=np.nan,**metrics(self.y,mean_p)))
                mc_pred=frame.mc_pred.to_numpy()
                strategies.extend([('B','mc_mean',mc_pred,frame.mc_confidence.to_numpy()),('C','mc_mean',mc_pred,-frame.MI.to_numpy())])
            frames.append(frame)
            for strategy,prediction,pred,score in strategies:
                rv,cv,rk=self.review(self.y,pred,score,kind,seed,sigma,strategy,prediction)
                reviews.append(rv);curves.append(cv);rankings.append(rk)
            print(kind,seed,'sigma',sigma,'完成',flush=True)
        model.eval()
        assert all(torch.equal(v.detach().cpu(),weights[k]) for k,v in model.state_dict().items())
        assert sha(cp)==r['sha256']
        tables=dict(predictions=pd.concat(frames,ignore_index=True),classification=pd.DataFrame(class_rows),review=pd.concat(reviews,ignore_index=True),curves=pd.concat(curves,ignore_index=True),rankings=pd.concat(rankings,ignore_index=True))
        for name,table in tables.items():csv(out/f'{name}.csv',table)
        if modes:csv(out/'module_modes.csv',pd.DataFrame(modes))
        if mi_checks:csv(out/'MI_checks.csv',pd.DataFrame(mi_checks))
        self.verify_evaluation(out)
        self.verify_more(out)
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for ax,sigma in zip(axes.flat,SIGMAS):
            for strategy in sorted(tables['curves'].strategy.unique()):
                g=tables['curves'].query('sigma==@sigma and strategy==@strategy')
                ax.plot(g.coverage,g.retained_risk,label=strategy)
                ax.axhline(g.theoretical_random_risk.iloc[0],linestyle='--',alpha=.4,label=f'{strategy} theoretical random')
            ax.set(xlim=(1,0),ylim=(0,1),xlabel='Coverage 1 -> 0 (more review ->)',ylabel='Retained risk [0,1]',title=f'{kind} seed{seed}, sigma={sigma:g}');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'risk_coverage.png',dpi=180);plt.close(fig)
        write_json(out/'completed.json',dict(passed=True,seconds=time.perf_counter()-began,N=self.n,weights_unchanged=True,checkpoint_unchanged=True,at_utc=utc()))
        return out

    def verify_more(self,out):
        ps=pd.read_csv(out/'predictions.csv',float_precision='round_trip');rv=pd.read_csv(out/'review.csv',float_precision='round_trip')
        rk=pd.read_csv(out/'rankings.csv',float_precision='round_trip');cv=pd.read_csv(out/'curves.csv',float_precision='round_trip')
        assert len(ps)==4*self.n and not ps.isna().any().any()
        for sigma in SIGMAS:
            g=ps.query('sigma==@sigma').sort_values('test_index')
            assert len(g)==self.n and np.array_equal(g.true_label,self.y)
            choices=[('A',g.deterministic_pred.to_numpy(),np.maximum(g.deterministic_p,1-g.deterministic_p))]
            if 'mean_p' in g:
                draws=g[[f'p_t{i+1:02d}' for i in range(30)]].to_numpy()
                assert np.isfinite(draws).all() and ((draws>=0)&(draws<=1)).all()
                assert np.allclose(entropy(g.mean_p),g.predictive_entropy,atol=1e-12,rtol=1e-12)
                assert np.allclose(entropy(draws).mean(1),g.expected_entropy,atol=1e-12,rtol=1e-12)
                assert (g.MI_raw>=-self.MI_TOL).all()
                choices += [('B',g.mc_pred.to_numpy(),np.maximum(g.mean_p,1-g.mean_p)),('C',g.mc_pred.to_numpy(),-g.MI)]
            for strategy,pred,score in choices:
                a,b,c=self.review(self.y,pred,np.asarray(score),g.model.iloc[0],int(g.seed.iloc[0]),sigma,strategy,'deterministic' if strategy=='A' else 'mc_mean')
                old=rv.query('sigma==@sigma and strategy==@strategy').reset_index(drop=True)
                for col in a.select_dtypes(include='number').columns:assert np.allclose(a[col],old[col],atol=1e-12,rtol=1e-12,equal_nan=True),col
                rank=rk.query('sigma==@sigma and strategy==@strategy').sort_values('test_index')
                assert np.array_equal(rank.review_rank,c.review_rank) and np.array_equal(rank.tie_rank,self.tie_rank)
                assert np.array_equal(rank.predicted_label,pred) and np.array_equal(rank.true_label,self.y)
                curve=cv.query('sigma==@sigma and strategy==@strategy').sort_values('retained_n')
                assert np.allclose(curve.coverage,curve.retained_n/self.n,atol=1e-12) and (curve.coverage>0).all()
                assert np.allclose(curve.theoretical_random_risk,(pred!=self.y).mean(),atol=1e-12)
        return True

    def aggregate(self):
        cl=pd.concat([pd.read_csv(p/'classification.csv') for p in self.outputs],ignore_index=True)
        rv=pd.concat([pd.read_csv(p/'review.csv') for p in self.outputs],ignore_index=True)
        cv=pd.concat([pd.read_csv(p/'curves.csv') for p in self.outputs],ignore_index=True)
        assert len(cl)==48 and len(rv)==300 and len(cv)==60*self.n
        r20=rv.query('target_review_pct==20').copy();assert len(r20)==60
        m=self.n*20//100;assert (r20.reviewed_n==m).all() and (r20.retained_n==self.n-m).all()
        assert np.allclose(r20.retained_risk,(r20.FP+r20.FN)/(self.n-m))
        classkeys=['model','seed','sigma','prediction']
        main=r20.merge(cl,on=classkeys,suffixes=('_retained','_classification'),validate='many_to_one')
        assert len(main)==60
        cm=['AUROC','AP','accuracy','sensitivity','specificity','TN','FP','FN','TP']
        rm=['retained_risk','coverage','FP','FN','removed_error_fraction','referred_pneumonia_fraction','theoretical_random_risk']
        cstats=cl.groupby(['model','prediction','sigma'])[cm].agg(['mean','std','count'])
        rstats=r20.groupby(['model','strategy','prediction','sigma'])[rm].agg(['mean','std','count'])
        for table in (cstats,rstats):
            table.columns=['_'.join(c) for c in table.columns]
        cstats=cstats.reset_index();rstats=rstats.reset_index()
        assert (cstats.AUROC_count==3).all() and (rstats.retained_risk_count==3).all()
        d=r20.query('model=="dropout" and strategy in ["B","C"]').pivot(index=['seed','sigma'],columns='strategy',values='retained_risk').reset_index()
        assert len(d)==12 and not d.isna().any().any()
        d['C_minus_B_retained_risk']=d.C-d.B;d['C_minus_B_percentage_points']=100*d.C_minus_B_retained_risk
        ds=d.groupby('sigma').agg(n_seeds=('seed','count'),mean_difference=('C_minus_B_retained_risk','mean'),sample_std_difference=('C_minus_B_retained_risk','std'),mean_percentage_points=('C_minus_B_percentage_points','mean'),sample_std_percentage_points=('C_minus_B_percentage_points','std')).reset_index()
        for sigma in SIGMAS:
            vals=d.query('sigma==@sigma').C_minus_B_retained_risk.to_numpy();row=ds.query('sigma==@sigma').iloc[0]
            assert len(vals)==3 and close(vals.mean(),row.mean_difference) and close(vals.std(ddof=1),row.sample_std_difference)
        outputs=dict(classification_per_seed=cl,review_all_per_seed=rv,review20_per_seed=r20,risk_coverage_full=cv,main_metrics_per_seed=main,classification_mean_sample_std=cstats,review20_mean_sample_std=rstats,dropout_C_minus_B_per_seed=d,dropout_C_minus_B_mean_sample_std=ds)
        for name,table in outputs.items():
            csv(self.out/f'{name}.csv',table)
            read=pd.read_csv(self.out/f'{name}.csv');assert len(read)==len(table)
            for col in table.select_dtypes(include='number').columns:assert np.allclose(read[col],table[col],atol=1e-12,rtol=1e-12,equal_nan=True)
        # Independent re-read of counts, primary formula, and all hashes.
        actual=load_json(self.out/'actual_test_data.json');assert actual['N']==self.n and sum(actual['labels'].values())==self.n
        rr=pd.read_csv(self.out/'review20_per_seed.csv');assert np.allclose(rr.retained_risk,(rr.FP+rr.FN)/rr.retained_n)
        assert np.allclose(rr.removed_error_fraction,(rr.original_errors-rr.FP-rr.FN)/rr.original_errors,equal_nan=True)
        final_hashes=[dict(path=r['path'],before=r['sha256'],after=sha(r['path']),unchanged=sha(r['path'])==r['sha256']) for r in self.records]
        assert all(r['unchanged'] for r in final_hashes);csv(self.out/'checkpoint_sha256_after.csv',pd.DataFrame(final_hashes))
        manifest=[dict(path=str(p.relative_to(self.out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(self.out.rglob('*')) if p.is_file()]
        csv(self.out/'result_file_manifest.csv',pd.DataFrame(manifest))
        lines=['# 最终测试评价实际结果',f'官方测试集N={self.n}；正常{int((self.y==0).sum())}，肺炎{int((self.y==1).sum())}。',f'目标20%复核实际转人工{m}，保留{self.n-m}，coverage={(self.n-m)/self.n:.10f}。','冻结方案在最终测试加载前保存；设置在查看验证结果后确定，不是研究开始前预注册。','## 主要结果：Dropout C-B（百分点）',ds.to_markdown(index=False),'正差值表示C高MI留下的错误风险高于B低置信度；负值表示低于。保留实际结果，不据此改设置或重跑。','逐seed明细见dropout_C_minus_B_per_seed.csv；分类与复核逐seed及均值/样本标准差分别保存。','三个seed共用同一测试集，不把样本重复当作独立患者；样本标准差仅反映训练/MC设置波动，不是泛化置信区间。不作显著性或临床安全性声明。','随机筛选参照为对应标签错误率的理论期望。零分母NaN；不假定人工判断全对，不计算模型+人工最终准确率。','这是人工模糊下的受控实验，不能代表真实医院数据变化。sigma=1.5高于模糊训练增强范围；冻结方案参考过验证结果。']
        with (self.out/'report_zh.md').open('x',encoding='utf-8') as f:f.write('\n\n'.join(lines))
        print(ds.to_string(index=False),flush=True)
