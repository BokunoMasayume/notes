收藏视频API



<https://api.bilibili.com/medialist/gateway/base/spaceDetail?media_id=377974029&pn=1&ps=20&keyword=&order=mtime&type=0&tid=0>



`media_id`: 收藏夹id

`pn`: 页数

`ps`: 每页行数



- `code`
- `message`: "success"
- `data`
  - `info`: 收藏夹的信息
    - `attr`
    - `cnt_info`
    - `cover`: 收藏夹封面
    - `cover_type`: 收藏夹状态，2公开，3私密
    - `ctime`
    - `fav_state`
    - `fid`
    - `id`: 收藏夹id
    - `intro`
    - `like_state`
    - `mid`: 所有用户mid
    - `state`
    - `title`: 收藏夹名称
    - `type`
    - `upper`: 收藏夹创建者信息
      - `face`: 头像
      - `followed`???
      - `mid`: id
      - `name`: 昵称
      - `vip_due_date`
      - `vip_pay_type`
      - `vip_statue`
      - `vip_type`
  - `medias`: 收藏视频==列表==
    - `attr`
    - `coin`
    - `cnt_info`: 体现视频影响力的一些数据
      - `coin`: 获得硬币数
      - `collect`: 收藏数
      - `danmaku`: 弹幕数
      - `play`: 播放数
      - `reply`: 回复数
      - `share`: 分享数
      - `thumb_down`
      - `thumb_up`: 赞数
    - `cover`: 封面
    - `ctime`
    - `duration`: 视频长度（秒
    - `fav_state`
    - `fav_time`
    - `id`: 视频AV号
    - `intro`: 视频简介
    - `like_state`
    - `link`
    - `page`: 分P数
    - `pubtime`
    - `tid`
    - `title`: 视频页标题
    - `type`
    - `pages`: 各分P信息==列表==
      - `duration`: 时长（秒
      - `from`
      - `id`: 视频CID
      - `metas`
      - `page`: 第几P
      - `title`: 分P标题
    - `upper`: 视频up主
      - `face`
      - `mid`
      - `name`
      - `vip_***`x4







返回的JSON对象

​	`data`: 返回数据容器

​		`info`: 收藏夹信息

​			`cover`:封面URL

​			`fid`

​			`id`: 收藏夹id

​			`mid`: 所有者id

​			`title`: 收藏夹名

​			`upper`: 更多所有者信息

​				`face`: 头像url

​				`name`: 昵称

​				`mid`: 用户id

​		`medias`: `收藏视频信息`列表

​		

收藏视频信息

​	`cnt_info`

​		`coin`: 

​		`collect`

​		`play`

​		`reply`

​		`share`

​		`thumb_down`

​		`thumb_up`

​		`danmaku`

​	`cover`: 封面

​	`id`: 视频ID？

​	`intro`: 视频简介

​	`link`: "bilibili://video/49488139"

​	`title`: 视频标题

​	`upper`: 上传者信息

​		`face`

​		`mid`

​		`name`

​	`page`： 分P数

​	`pages`: 分p信息列表

​		`id` ：

​		`title`： 分P标题

​		`page`：第几P

​		