视频信息API

<https://api.bilibili.com/x/web-interface/view/detail?aid=7858618>



`aid`: 视频AV号



- `code`
- `message`
- `ttl`
- `data`
  - `Card`: 视频主信息，见[用户信息API](./用户信息API.md)
  - `Related`: 相关视频信息列表
  - `Reply`: 一点不全又谜的回复信息
  - `Tags`: 视频tag信息
  - `View`:
    - `aid`: AV号
    - `attribute`
    - `cid`: 第一P视频号
    - `copyright`
    - `ctime`
    - `desc`: 视频简介
    - `dimension`
    - `duration`: 所有分P总时长（秒
    - `dynamic`
    - `no_cache`
    - `owner`: 所有者信息
      - `mid`
      - `face`
      - `name`
    - `pages`: 分页信息==列表==
      - `cid`: 视频号
      - `dimension`
      - `duration`: 时长
      - `from`
      - `page`: 第几P
      - `part`: 分P标题
      - `vid`
      - `weblink`
    - `pic`: 封面
    - `pubdate`
    - `rights`
    - `state`
    - `subtitle`
    - `tid`
    - `title`: 视频页标题
    - `tname`
    - `videos`: 总P数
    - `stat`: 视频影响力的一些数据‘
      - `aid`:
      - `coin`: 
      - `danmaku`
      - `dislike`
      - `favorite`
      - `his_rank`: 最高日排名,0是榜上无名
      - `like`
      - `now_rank`
      - `reply`
      - `share`
      - `view`