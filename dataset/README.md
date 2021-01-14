# Dataset

## Tranfer dataset between MacOS and Ubuntu

### MacOS

```bash
scp -rp ./dataset/ name@ip_address:/.../.../dataset
```

### Ubuntu

Install SSH and update its config
```bash
sudo apt-get install openssh-server -y

sudo nano /etc/ssh/sshd_config
```

Uncomment the following setting in the config
```
Port 22
```

Update the firewall
```bash
sudo ufw allow 22
sudo service ssh restart
```


