# H200\-qinghua服务器使用指南

有问题可以问管理员@Yiyang

## 1\. 快速开始

把下面内容加入本机 `~/.ssh/config`，把 `<username>` 换成你的服务器用户名：

```Plain Text
Host jump-h200-qinghua
  HostName 183.242.150.33
  Port 60022
  User <username>

Host <username>-h200-qinghua-1
  HostName 214.30.239.40
  User <username>
  ProxyJump jump-h200-qinghua

Host <username>-h200-qinghua-2
  HostName 214.30.239.42
  User <username>
  ProxyJump jump-h200-qinghua


```

常用登录：

```Bash
ssh jump-h200-qinghua
ssh <username>-h200-qinghua-1
ssh <username>-h200-qinghua-2


```

第一次登录后建议检查：

```Bash
whoami
pwd
ls -ld ~ /data /data/shared
conda --version
uv --version
nvidia-smi


```

## 2\. 账号与 SSH

### 自己新增设备公钥

在新设备上查看公钥：

```Bash
cat ~/.ssh/id_ed25519.pub

```

在旧设备上执行，把 `NEW_KEY` 换成新设备输出的那一整行公钥：

```Bash
read -r NEW_KEY <<'EOF'
ssh-ed25519 <新设备公钥内容>
EOF

printf '%s\n' "$NEW_KEY" | ssh-keygen -lf -

for host in jump-h200-qinghua <username>-h200-qinghua-1 <username>-h200-qinghua-2; do
  ssh "$host" 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
  printf '%s\n' "$NEW_KEY" | ssh "$host" 'read -r key; grep -qxF "$key" ~/.ssh/authorized_keys || printf "%s\n" "$key" >> ~/.ssh/authorized_keys'
done


```

完成后在新设备上测试：

```Bash
ssh jump-h200-qinghua
ssh <username>-h200-qinghua-1
ssh <username>-h200-qinghua-2


```

如果旧设备已经完全无法登录，再联系管理员追加公钥。

### GitHub

GitHub key、Git 用户名、邮箱和 token 只配置在你自己的跳板机账号下。H200 不直接访问 GitHub；需要拉代码时，优先在跳板机下载到对应 H200 的挂载盘。

```Bash
ssh jump-h200-qinghua
cd /data-214-30-239-40/home/$USER/projects
git clone <repo-url>


```

## 3\. 路径与存储

两台 H200：

```Plain Text
H200-1: 214.30.239.40
H200-2: 214.30.239.42
每台: 8 x NVIDIA H200
每台: /data 为本机数据盘


```

H200 上常用路径：

```Plain Text
/home/<username>/         home 目录
/data/home/<username>/    home 实际位置
/home/<username>/shared/  指向 /data/shared
/data/shared/             本机共享目录
/data/cache/              本机缓存根目录
/data/tmp/                临时目录


```

跳板机通过 NFS 挂载两台 H200 的 `/data`：

```Plain Text
跳板机 /data-214-30-239-40  -> H200-1 /data
跳板机 /data-214-30-239-42  -> H200-2 /data


```

两台 H200 的 `/data` 不互通。放到 H200\-1 的文件，H200\-2 看不到；两台都要用的文件，需要分别放到两边。

## 4\. 软件环境

H200 可以使用本地已有工具和系统级内网源。不要在 H200 上配置公网源、代理、VPN、Clash、mihomo、tinyproxy、GitHub token 或 Hugging Face 公网 endpoint。已安装工具包括：

```Plain
CUDA / NVIDIA driver
Docker / NVIDIA runtime
Python / conda / mamba / uv
Node.js / npm
Go / Rust
zsh / tmux / vim / jq / rsync
gcc / g++ / make / cmake / pkg-config / python3-dev / linux-libc-dev / linux-headers-$(uname -r) / ffmpeg


```

系统级内网源：

```Plain Text
pip:   http://214.30.239.5/simple/
uv:    http://214.30.239.5/simple/
npm:   http://214.30.239.5:82/
APT:   http://214.30.239.5:84/ubuntu

conda:
  http://214.30.239.5:85/main
  http://214.30.239.5:85/r
  http://214.30.239.5:85/free
  http://214.30.239.5:86/main
  http://214.30.239.5:86/r
  http://214.30.239.5:86/free


```

conda 当前只确认有 `main`、`r`、`free` 内网 channel；没有确认 `conda-forge`、`pytorch`、`nvidia`、`bioconda`、`rapidsai`。如果 conda 找不到包，优先改用 pip/uv 内网源或离线包。常用检查：

```Bash
python3 -m pip config list
conda config --show-sources
npm config get registry
echo "$UV_INDEX_URL"
uv pip install --dry-run --system six


```

### conda

H200 自身已安装 `/opt/miniconda3`，支持：

```Bash
conda env list
conda activate <env>


```

如果旧会话里 `conda activate` 异常，重新登录，或执行：

```Bash
exec bash
# 或
exec zsh


```

跳板机也有共享 Miniconda 和基础编译工具（gcc / g\+\+ / make / build\-essential / python3\-dev / linux\-libc\-dev / linux\-headers\-$\(uname \-r\)），适合下载源码、wheel、conda 包、npm 包、模型或数据，并处理需要本机编译的 Python 包。不要在跳板机直接创建准备给 H200 运行的 conda 环境，因为跳板机看到的是 `/data-214-30-239-40/...`，H200 看到的是 `/data/...`，绝对路径不同。

## 5\. 文件传输

本机和 H200 之间：

```Bash
scp local-file <username>-h200-qinghua-1:/data/tmp/
scp <username>-h200-qinghua-1:/data/tmp/remote-file .


```

两台 H200 之间：

```Bash
ssh <username>-h200-qinghua-1
scp some-file h200-qinghua-2:/data/tmp/


```

从跳板机向 H200 挂载盘写文件：

```Bash
ssh jump-h200-qinghua
cd /data-214-30-239-40/home/$USER/projects


```

## 6\. 依赖安装与离线包

优先顺序：

```Plain Text
1. H200 上使用系统级内网源。
2. 内网源缺包或缺版本时，在跳板机下载/打包，写入对应 H200 的挂载盘。
3. H200 上从 /data/shared/offline 安装。
4. 不要临时添加公网源或代理。


```

离线目录：

```Plain Text
跳板机:
/data-214-30-239-40/shared/offline
/data-214-30-239-42/shared/offline

H200:
/data/shared/offline


```

推荐目录结构：

```Plain Text
/data/shared/offline/
  wheels/        pip/uv 使用的 .whl、requirements.txt、constraints.txt
  conda/         conda 包、conda-pack 环境包、environment.yml
  npm/           npm tarball、package-lock.json、node_modules 压缩包
  uv/            uv lock、requirements、wheelhouse 或项目缓存
  docker/        docker save 导出的镜像 tar / tar.zst
  apt/           deb 包及依赖集合
  models/        模型文件
  datasets/      数据集文件
  source/        源码包、release tarball、二进制工具
  manifests/     每批离线包的说明和校验结果


```

### pip / uv

H200 上优先直接使用内网源：

```Bash
pip install -r requirements.txt
uv pip install -r requirements.txt


```

缺包时，在跳板机准备 wheelhouse。H200\-1：

```Bash
OFF=/data-214-30-239-40/shared/offline/wheels/<project-or-env>
mkdir -p "$OFF"
python3 -m pip download -r requirements.txt -d "$OFF"
cp requirements.txt "$OFF/"
sha256sum "$OFF"/* > "$OFF/SHA256SUMS"


```

H200\-2：

```Bash
OFF=/data-214-30-239-42/shared/offline/wheels/<project-or-env>
mkdir -p "$OFF"
python3 -m pip download -r requirements.txt -d "$OFF"
cp requirements.txt "$OFF/"
sha256sum "$OFF"/* > "$OFF/SHA256SUMS"


```

在对应 H200 上安装：

```Bash
pip install --no-index --find-links /data/shared/offline/wheels/<project-or-env> -r requirements.txt
uv pip install --no-index --find-links /data/shared/offline/wheels/<project-or-env> -r requirements.txt


```

包含 CUDA、PyTorch、flash\-attn、xformers 等二进制 wheel 时，必须确认 Python 版本、CUDA 版本和 Linux x86\_64 架构匹配。

### conda / mamba

如果 H200 内网 conda channel 找不到包，不要加公网 channel。推荐在跳板机或兼容 Linux x86\_64 环境中打包完整环境。

```Bash
OFF=/data-214-30-239-40/shared/offline/conda/<env>
mkdir -p "$OFF"
conda activate <env>
conda-pack -n <env> -o "$OFF/<env>.tar.gz"
conda env export --from-history > "$OFF/environment.from-history.yml"
sha256sum "$OFF"/* > "$OFF/SHA256SUMS"


```

在 H200 上解包：

```Bash
mkdir -p /data/cache/conda/envs/$USER/<env>
tar -xzf /data/shared/offline/conda/<env>/<env>.tar.gz -C /data/cache/conda/envs/$USER/<env>
/data/cache/conda/envs/$USER/<env>/bin/conda-unpack


```

### npm / Node\.js

H200 优先使用内网 npm registry：

```Bash
npm ci


```

如果缺包，在跳板机准备完整依赖：

```Bash
OFF=/data-214-30-239-40/shared/offline/npm/<project>
mkdir -p "$OFF"
npm ci
tar -czf "$OFF/node_modules.tar.gz" node_modules package-lock.json
sha256sum "$OFF"/* > "$OFF/SHA256SUMS"


```

在 H200 上解包：

```Bash
tar -xzf /data/shared/offline/npm/<project>/node_modules.tar.gz


```

### Docker

H200 不直接联网拉镜像。先在跳板机或其他可联网机器上保存镜像，再放到目标 H200 的离线目录。

```Bash
OFF=/data-214-30-239-40/shared/offline/docker/<image-name>
mkdir -p "$OFF"
docker pull <image:tag>
docker save <image:tag> | zstd -T0 -o "$OFF/<tag>.tar.zst"
sha256sum "$OFF"/* > "$OFF/SHA256SUMS"


```

在 H200 加载：

```Bash
zstd -dc /data/shared/offline/docker/<image-name>/<tag>.tar.zst | docker load
docker images


```

### 模型和数据

模型和数据尽量保持原始目录结构：

```Plain Text
/data/shared/offline/models/<model-name>/<revision-or-original-dir>/
/data/shared/offline/datasets/<dataset-name>/<version-or-original-dir>/


```

代码中使用本地路径。不要在 H200 上执行在线 `hf download`，也不要写入公网/代理 `HF_ENDPOINT`。

## 7\. VS Code Remote\-SSH

H200 不能访问 VS Code Marketplace、公网 CDN 或跳板机代理。使用 VS Code Remote\-SSH 时，不要让 VS Code 在 SSH 握手阶段自动安装扩展。本地 VS Code `settings.json` 建议：

```JSON
"remote.SSH.defaultExtensions": [],
"remote.SSH.remotePlatform": {
  "<your-host>": "linux"
}


```

离线安装扩展流程：

```Bash
# 本地有网机器
mkdir -p /tmp/vscode-vsix
# 下载对应扩展的 VSIX 到 /tmp/vscode-vsix

ssh <your-host> 'mkdir -p ~/vscode-remote-vsix'
scp /tmp/vscode-vsix/*.vsix <your-host>:~/vscode-remote-vsix/

ssh <your-host> '
code_server=$(find ~/.vscode-server/cli/servers -path "*/server/bin/code-server" -type f | sort | tail -1)
for f in ~/vscode-remote-vsix/*.vsix; do
  "$code_server" --install-extension "$f" --force --do-not-include-pack-dependencies
done
"$code_server" --list-extensions --show-versions
'


```

关键参数是 `--do-not-include-pack-dependencies`。如果 `code-server` 路径为空，先用 VS Code Remote\-SSH 成功连接一次，让远端生成 `~/.vscode-server/cli/servers/.../server/bin/code-server`。

## 8\. 双机训练

两台 H200 可以通过 InfiniBand 做双机训练。控制面使用管理 IP，NCCL 数据通信走 IB/RDMA。训练前建议两台都检查：

```Bash
ibstat
ibdev2netdev
rdma link show
nvidia-smi topo -m


```

常用环境变量：

```Bash
export MASTER_ADDR=214.30.239.40
export MASTER_PORT=29500

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_SOCKET_IFNAME=bond0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


```

PyTorch `torchrun` 示例。H200\-1：

```Bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=214.30.239.40 \
  --master_port=29500 \
  train.py <args>


```

H200\-2：

```Bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=214.30.239.40 \
  --master_port=29500 \
  train.py <args>


```

DeepSpeed hostfile：

```Plain Text
214.30.239.40 slots=8
214.30.239.42 slots=8


```

启动：

```Bash
deepspeed \
  --hostfile hostfile \
  --master_addr 214.30.239.40 \
  --master_port 29500 \
  train.py <args>


```

## 9\. 常见问题

**没有权限写目录**确认你用自己的跳板机账号和自己的 H200 账号登录。个人项目应写到：

```Plain Text
/data-214-30-239-40/home/$USER/...
/data-214-30-239-42/home/$USER/...


```

**H200 上不能下载 GitHub / Hugging Face / Docker 镜像**这是预期行为。H200 不直接访问公网，也不使用跳板机代理。到跳板机或其他有网机器下载后，放入对应 H200 的 NFS 挂载盘。**conda 找不到包**不要添加公网 channel。优先尝试 pip/uv 内网源，或走离线包/conda\-pack。**VS Code Remote\-SSH 卡住**先确认本地 `remote.SSH.defaultExtensions` 为空，避免连接时自动安装扩展。需要扩展时按离线 VSIX 流程安装。**双机训练卡在初始化**确认两台都启动、`node_rank` 分别是 0 和 1、`MASTER_ADDR=214.30.239.40`、`MASTER_PORT` 一致，且 H200\-1 能免密 SSH 到 H200\-2。

> (Note: The content is generated by AI. Please use with caution.)
