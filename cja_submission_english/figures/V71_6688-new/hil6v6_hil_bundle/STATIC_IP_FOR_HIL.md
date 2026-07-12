# 6v6 HIL 联动建议静态IP配置

目的是把“本机（Linux）—NX — 本机”的链路固定到同一网段，避免端口和DHCP漂移导致中断。

## 1）本机静态IP（Linux示例）

假设接到交换机的接口是 `enp3s0`，目标网段 `192.168.1.0/24`，本机给 `192.168.1.100`，NX给 `192.168.1.20`（你可改成自己想要的IP）。

```bash
sudo ip addr flush dev enp3s0
sudo ip addr add 192.168.1.100/24 dev enp3s0
sudo ip link set enp3s0 up
```

可在重启后永久生效，请改成你的网络管理方式（`nmcli`/`netplan`）：

```bash
sudo ip route add default via 192.168.1.1 dev enp3s0  # 若有网关可加；无网关可省略
```

检查：

```bash
ip a show enp3s0
ip route
ping -c 3 192.168.1.20
```

## 2）NX静态IP（Linux/Ubuntu NX）

```bash
sudo ip addr flush dev eth0
sudo ip addr add 192.168.1.20/24 dev eth0
sudo ip link set eth0 up
sudo ip route add default via 192.168.1.100 dev eth0
```

确认连接：

```bash
ip a
ping -c 3 192.168.1.100
```

## 3）HIL连接链路检查

只要本机和NX都在同一网段且防火墙放通TCP 5500，本机会连上：

```bash
nc -vz 192.168.1.20 5500
```

脚本启动顺序：
1. 本机先启动服务端（端口5500）  
2. 本机启动 5 个本地policy节点（agent 1-5）  
3. NX启动1个policy节点（agent 0）

建议先用上面顺序跑一次成功通路，再考虑长期替换为自动化脚本。
