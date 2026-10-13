import argparse,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from fastwam.models.wan22.helpers.loader import _load_registered_model,_resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam_c2r.data import text_key
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prep",required=True)
    parser.add_argument("--cache",required=True)
    args=parser.parse_args()
    prep=Path(args.prep)
    rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"]); local=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dist.init_process_group("nccl")
    prompts=json.loads((prep/"prompts.json").read_text())
    cache=Path(args.cache); cache.mkdir(parents=True,exist_ok=True)
    todo=[p for p in prompts[rank::world] if not (cache/(text_key(p)+".pt")).exists()]
    print("TEXT_CACHE",rank,"todo",len(todo),flush=True)
    if todo:
        _,text,_,tokenizer=_resolve_configs("Wan-AI/Wan2.2-TI2V-5B","Wan-AI/Wan2.2-TI2V-5B",True)
        text.download_if_necessary();tokenizer.download_if_necessary()
        model=_load_registered_model(text.path,"wan_video_text_encoder",torch.bfloat16,f"cuda:{local}")
        model.eval().requires_grad_(False)
        tokenizer=HuggingfaceTokenizer(tokenizer.path,seq_len=512,clean="whitespace")
        with torch.no_grad():
            for start in range(0,len(todo),16):
                batch=todo[start:start+16]
                ids,mask=tokenizer(batch,return_mask=True,add_special_tokens=True)
                context=model(ids.cuda(),mask.cuda().bool())
                for i,prompt in enumerate(batch):
                    n=int(mask[i].sum())
                    payload={"context":context[i,:n].cpu().contiguous(),"context_len":512,"prompt":prompt,
                        "encoder_revision":"921dbaf3f1674a56f47e83fb80a34bac8a8f203e"}
                    dst=cache/(text_key(prompt)+".pt");tmp=dst.with_suffix(f".r{rank}.tmp")
                    torch.save(payload,tmp);tmp.replace(dst)
                if start%160==0: print("TEXT_CACHE_PROGRESS",rank,start,len(todo),flush=True)
    dist.barrier()
    if rank==0:
        assert all((cache/(text_key(p)+".pt")).exists() for p in prompts)
        (prep/"text_cache.json").write_text(json.dumps({"status":"passed","prompts":len(prompts),"context_len":512,"storage":"valid tokens only; exact zero-padding restored at load"})+"\n")
    dist.destroy_process_group()
if __name__=="__main__":main()

