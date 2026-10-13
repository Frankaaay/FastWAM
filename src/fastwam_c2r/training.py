import json
import math
from pathlib import Path
import shutil
import time
import torch
from torch.utils.data import DataLoader
from fastwam.trainer import Wan22Trainer
from .data import C2RSampler

class C2RTrainer(Wan22Trainer):
    def _save_trainer_state(self,state_path):
        super()._save_trainer_state(state_path)
        state=Path(state_path)
        manifest=json.loads(Path(self.cfg.c2r.dataset.manifest).read_text())
        model = self.accelerator.unwrap_model(self.model)
        (state/"tokenizer.json").write_text(json.dumps(getattr(model.vae, "binding", None))+"\n")
        (state/"normalization.json").write_text(json.dumps(manifest["normalization"],indent=2)+"\n")
        shutil.copyfile(Path(self.output_dir)/"config.yaml",state/"config.yaml")

    def load_training_state(self, state_dir):
        model = self.accelerator.unwrap_model(self.model)
        saved = json.loads((Path(state_dir)/"tokenizer.json").read_text())
        if saved != getattr(model.vae, "binding", None):
            raise ValueError("Full-state tokenizer binding mismatch")
        super().load_training_state(state_dir)

    def _build_loader(self,dataset,worker_init_fn=None):
        self.train_sampler=C2RSampler(dataset,self.seed,self.batch_size,self.accelerator.num_processes)
        return DataLoader(dataset,batch_size=self.batch_size,sampler=self.train_sampler,
            num_workers=self.num_workers,pin_memory=True,worker_init_fn=worker_init_fn,
            multiprocessing_context="spawn" if self.num_workers else None,
            **({"prefetch_factor":2} if self.num_workers else {}))
    def _build_scheduler(self,scheduler_type,total_train_steps,warmup_steps=0):
        table=json.loads(Path(self.cfg.c2r.lr_table).read_text())
        self.lr_factors=table["lr_factors"]
        assert len(self.lr_factors)==int(self.cfg.c2r.total_steps)+1
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step:self.lr_factors[min(step,len(self.lr_factors)-1)])
    def _wandb_log(self,payload):
        if "train/loss" not in payload: return
        payload["performance/samples_per_sec"] *= self.gradient_accumulation_steps
        if not all(math.isfinite(float(v)) for v in payload.values()): raise FloatingPointError(payload)
        expected=self.learning_rate*self.lr_factors[self.global_step]
        if not math.isclose(payload["train/lr"],expected,rel_tol=1e-7,abs_tol=1e-14):
            raise AssertionError((self.global_step,payload["train/lr"],expected))
        engine_steps=getattr(self.model,"global_steps",None)
        if engine_steps is not None and int(engine_steps)!=self.global_step:
            raise AssertionError(f"DeepSpeed update mismatch: engine={engine_steps}, trainer={self.global_step}")
        payload={**payload,"step":self.global_step,"epoch":self.epoch,"batch_in_epoch":self.batch_in_epoch,
                 "wall_time":time.time(),"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                 "engine_steps":engine_steps,"loss_scope":"last_microbatch_rank_average"}
        with (Path(self.output_dir)/"metrics.jsonl").open("a") as f: f.write(json.dumps(payload)+"\n")
        (Path(self.output_dir)/"progress.json.tmp").write_text(json.dumps(payload,indent=2)+"\n")
        (Path(self.output_dir)/"progress.json.tmp").replace(Path(self.output_dir)/"progress.json")
        super()._wandb_log(payload)
    def save_checkpoint(self):
        # The base loop requests a save twice at the final epoch boundary.
        if getattr(self, "_saved_step", None) == self.global_step:
            return self._saved_paths
        paths=super().save_checkpoint()
        if self.accelerator.is_main_process:
            state=Path(paths["state_path"])
            report={"step":self.global_step,"epoch":self.epoch,"batch_in_epoch":self.batch_in_epoch,
                    "world_size":self.accelerator.num_processes,"batch_size":self.batch_size,
                    "gradient_accumulation_steps":self.gradient_accumulation_steps,
                    "lr":self.optimizer.param_groups[0]["lr"],"weights":str(paths["weights_path"])}
            (state/"c2r_complete.json").write_text(json.dumps(report,indent=2)+"\n")
        self.accelerator.wait_for_everyone()
        self._saved_step, self._saved_paths = self.global_step, paths
        return paths

