# Docker 环境搭建

## 安装核心组件

- docker: 命令行工具（客户端）
- colima: 容器运行时

```bash
brew install docker colima
```

## 启动 Docker 环境 & 验证安装

安装完成后，只需在终端输入：

```bash
colima start

docker ps
```

> [!NOTE]
> 注意： 第一次启动会下载一个轻量级的镜像并配置虚拟机，可能需要几分钟。默认情况下，它会分配 2个 CPU 和 2GB 内存。

- 停止环境： `colima stop`
- 查看状态： `colima status`

> [!TIP]
> 由于 Docker Desktop 通常会接管 /usr/local/bin/docker 这个位置，Homebrew 为了防止冲突，默认不会强制链接（link）它。
>
> 1. 强制链接 Docker 客户端: `brew link --overwrite docker`
> 2. 验证链接是否成功: `which docker` => `/usr/local/bin/docker` 或 `/opt/homebrew/bin/docker`

## 运行 Postgres 容器

```bash
docker run -d \
--name postgres-test \
-e POSTGRES_PASSWORD=postgres \
-p 5432:5432 \
postgres:latest
```
