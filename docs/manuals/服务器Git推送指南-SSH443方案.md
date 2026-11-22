# 服务器 → GitHub 推送实战记录（SSH-443 方案）

> 记录 2026-08 期间把训练产物（含 ~100 MB LFS 权重）从 SLURM 集群推送到 GitHub 的
> 完整踩坑过程与最终方案。**结论先行：HTTPS 推送在该集群上不可靠，改用 SSH 443 后一次通过。**
>
> 仓库：`git@github.com:Xpc1489588006/rcr-yolo.git`

> **当前项目约定（2026-10-27）：服务器统一在 `~/rcr/rcr_yolo_c0ffed2` 使用 `c0ffed2` 代码。** 下文记录的 `~/rcr/rcr_yolo` 与 `origin/main` 操作是历史流程，不再作为当前工作指令；不得在统一工作目录执行 `git pull origin main`、`git reset --hard origin/main` 或将其推送到 `main`。

---

## 1. TL;DR（只看这一节也够）

服务器推送代码/训练产物，只需：

```bash
cd ~/rcr/rcr_yolo_c0ffed2
git add -A && git commit -m "..."
git push origin <已确认目标分支>  # 仅在确认代码基线一致后推送
```

前提是已按 §4 完成一次性配置（当前机器已配好）。

---

## 2. 故障现象全记录（HTTPS 阶段）

按时间顺序，HTTPS 推送依次出现过以下故障：

| # | 现象 | 原因 | 当时的处理 |
|---|---|---|---|
| 1 | `Password authentication is not supported` | GitHub 2021 起禁用账号密码 | 改用 PAT（Tokens classic，scope=repo） |
| 2 | `conda install git-lfs` 卡在 `Platform: linux-64` | conda-forge 官方源 + SUSTech 镜像 + pytorch + defaults 多源混排，依赖求解卡死 | `--override-channels` 只用单一镜像源 |
| 3 | `Failed to connect to github.com port 443 after 130435 ms` | 集群出口到 GitHub 连接时通时不通 | `http.connectTimeout 15` 快速失败 + 反复重试 |
| 4 | `Remote does not support the Git LFS locking API` + `dial tcp ... i/o timeout` | 新版 git-lfs 推送前做锁校验，GitHub 不支持该 API，请求又超时导致整个推送中止 | `git config --global lfs.locksverify false` |
| 5 | **`RPC failed; HTTP 408` + `unexpected disconnect while reading sideband packet`**（反复出现） | 对象包 100% 写完后、服务端确认引用更新阶段被切断。疑似集群出口代理缓冲大 POST 请求、慢速转发导致 GitHub 侧超时 | 见 §3——所有 HTTP 调优均无效 |

### 故障 5 的关键特征（为什么判定不是偶发）

- 每次都**死在同一个位置**：`Writing objects: 100% ... done` 之后、`main -> main` 之前；
- LFS 权重上传（独立通道）**每次都成功**（慢速 160–530 KB/s 也能传完 100 MB）；
- 以下配置全部就位后依然复现：

```
http.version=HTTP/1.1        # 治 "unexpected disconnect" 的常见方案，此处无效
http.postbuffer=524288000
http.connecttimeout=15
http.lowspeedlimit=1000
http.lowspeedtime=30
```

→ 结论：问题在链路的代理/网关层，客户端无法修复，**必须换通道**。

---

## 3. 最终方案：SSH over 443

### 3.1 为什么有效

- SSH 走独立的协议通道，不经过缓冲/慢转 HTTPS 大请求的代理逻辑；
- 用 443 端口（`ssh.github.com:443`）而非默认 22，绕开可能被封的 22 端口；
- 实测 22.5 MiB 对象包以真实网速 196 KiB/s 稳稳传完，一次成功；
- 附带收益：**以后推送再也不用输 PAT**。

### 3.2 一次性配置步骤（新机器复现用）

```bash
# 1. 密钥（已有可跳过）
ls ~/.ssh/id_ed25519.pub 2>/dev/null || ssh-keygen -t ed25519 -C "slurm-cluster"

# 2. SSH 走 443
cat >> ~/.ssh/config << 'EOF'
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
EOF
chmod 600 ~/.ssh/config

# 3. 公钥（~/.ssh/id_ed25519.pub 内容）加到 GitHub：
#    Settings → SSH and GPG keys → New SSH key
#    ⚠️ 必须加到对仓库有写权限的账号（Xpc1489588006）

# 4. 验证指纹后接受主机密钥（以下为 GitHub 官方 ED25519 指纹，可核对）
#    SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU
ssh -T git@github.com
# 期望：Hi Xpc1489588006! You've successfully authenticated...

# 5. 切远程地址 + 关 LFS 锁校验
git config --global lfs.locksverify false
cd ~/rcr/rcr_yolo_c0ffed2
git remote set-url origin git@github.com:Xpc1489588006/rcr-yolo.git

# 6. 推送
# 推送前确认目标分支包含/基于 c0ffed2；不要推到 main
git push origin <已确认目标分支>
```

### 3.3 成功的输出长这样

```
Uploading LFS objects: 100% (4/4), 25 MB | ..., done.
Writing objects: 100% (63/63), 22.53 MiB | 196.00 KiB/s, done.
remote: Resolving deltas: 100% (4/4), completed with 2 local objects.
To github.com:Xpc1489588006/rcr-yolo.git
   05335aa..d297747  main -> main
```

---

## 4. 日常同步流程

### 4.1 服务器 → GitHub（推送训练产物）

```bash
cd ~/rcr/rcr_yolo_c0ffed2
git status                 # 先看清单，别把 datasets/ 混进去
git add -A
git commit -m "sync ablation runs"
# 推送前确认目标分支与 c0ffed2 基线一致；不要推到 main
git push origin <已确认目标分支>
git lfs ls-files | tail    # 抽查 .pt 已进 LFS
```

### 4.2 GitHub → 本地 Windows（拉取）

```powershell
cd d:\Desktop\paper\robot\pp\rcr_yolo
# 当前统一工作区固定在 c0ffed2；勿按历史流程拉取 origin/main。
git lfs ls-files                          # 验证
dir runs\rcr\ab1-orb\weights              # .pt 应为 5-7 MB 实体
# 若某 .pt 只有 ~130 字节（指针未展开）：
git lfs pull
```

本地推送遇代理报错时：`git -c http.proxy= -c https.proxy= push origin main`。

### 4.3 服务器 ← GitHub（拉取本地改动）

```bash
cd ~/rcr/rcr_yolo_c0ffed2
git status                       # 训练会不断改写 runs/ 下被追踪的文件
# 不要 checkout/reset runs/ 或 pull origin/main；先保留训练产物并确认目标代码提交。
```

---

## 5. 同步纪律（训练期间的铁律）

1. **正在训练的 run 目录两个方向都不要同步**：
   - `results.csv`/`last.pt` 每个 epoch 都在重写，推上去是半成品；
   - 推半成品权重到 `runs/rcr/<name>/weights/best.pt` 会让 `ablation.sh`
     的"已存在 best.pt 则跳过"逻辑**误判该变体已完成**；
2. **集群曾发生 I/O 错误导致训练中断**，崩溃留下的不完整 `best.pt` 同样触发误跳过。
   补跑前先删目录：`rm -rf runs/rcr/<变体名>` 再 `sbatch slurm/ablation.sh`；
3. 推送/写论文前用行数核对每个变体的完整性（151 行 = 1 表头 + 150 epoch）：

   ```bash
   for d in runs/rcr/*/; do
     echo "$(basename $d): $(wc -l < $d/results.csv 2>/dev/null || echo 0) 行"
   done
   ```

---

## 6. 排障速查表

| 报错 | 处理 |
|---|---|
| `Permission denied (publickey)` | 公钥未加 / 加错账号；`ssh -T git@github.com` 验证 |
| `ssh: connect to host ssh.github.com port 443 ... timed out` | 443 SSH 也被封（本项目未遇到）；回退用 git bundle 借道本地推送 |
| `Remote does not support the Git LFS locking API` | `git config --global lfs.locksverify false` |
| 本地 `.pt` 只有 130 字节 | `git lfs pull` |
| 本地 pull 报代理错误 | `git -c http.proxy= -c https.proxy= pull origin main` |
| 服务器 pull 被 `runs/` 脏文件阻挡 | `git checkout -- runs/`（或先提交服务器侧改动） |

### 借道本地推送（终极保底）

```bash
# 服务器：把未推送的提交打包
cd ~/rcr
git -C rcr_yolo bundle create rcr_pending.bundle main --not origin/main
# 用 scp/平台文件传输把 bundle 弄到本地
```

```powershell
# 本地：取回提交并代推
cd d:\Desktop\paper\robot\pp\rcr_yolo
git fetch D:\path\to\rcr_pending.bundle main:server-main
git merge server-main
git push origin main
```

---

## 7. 当前生效的配置清单

**服务器（`~/.gitconfig`）**：

```
http.sslverify=false          # 集群证书环境遗留，SSH 方案下已无关紧要
http.postbuffer=524288000
http.lowspeedlimit=1000
http.lowspeedtime=30
http.version=HTTP/1.1
http.connecttimeout=15
lfs.locksverify=false
credential.helper=store       # HTTPS 时代遗留；SSH 推送不再需要
```

**仓库 remote**：`origin = git@github.com:Xpc1489588006/rcr-yolo.git`（SSH）

**`.gitattributes` 关键规则**：`*.pt` 走 LFS；`*.sh` 强制 LF；csv/json/yaml 统一 LF。

---

*最后更新：2026-08-27，SSH 方案验证通过（05335aa..d297747）。*
