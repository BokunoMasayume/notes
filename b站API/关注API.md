关注API

<https://api.bilibili.com/x/relation/followings?vmid=8796848&pn=1&ps=2&order=desc>

`vmid`: 关注者mid

`pn`: 分页数

`ps`：每页行数

`order`：？？？



- `code`
- `message`
- `ttl`
- `data`
  - `total`: 关注总数
  - `re_version`: 
  - `list`: 关注者信息列表
    - `attribute`
    - `mtime`
    - `official_verify`
    - `face`: 头像
    - `mid`: 用户id
    - `sign`: 个人签名
    - `special`
    - `tag`
    - `uname`: 昵称
    - `vip`









返回的json对象中：

`data`:

​	`total`: 总关注数

​	`list`: 关注用户列表





关注用户对象：

​	`face`: 头像URL

​	`mid`: mid

​	`official_verify`:

​			`type`:

​			`sign`:描述

​	`uname`:昵称

​	`vip`:就那个对象