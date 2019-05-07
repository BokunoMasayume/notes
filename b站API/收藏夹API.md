收藏夹API

<https://api.bilibili.com/x/space/fav/nav?mid=2009929>



- `code`
- `message`
- `ttl`
- `data`
  - `album`
  - `article`
  - `movie`
  - `playlist`
  - `topic`
  - `archive`: 收藏夹对象
    - `atten_count`
    - `cover`: 收藏夹中最新3个收藏信息
    - `ctime`
    - `cur_count`: 当前收藏数
    - `favoured`
    - `fid`
    - `max_count`
    - `media_id`: 收藏夹id
    - `mid`: 所有者id
    - `mtime`
    - `name`: 收藏夹名称
    - `state`: 已知2公开，3私密













返回的JSON对象

​	`data`

​		`archive`:收藏夹对象列表



收藏夹对象：

​	`name`: 收藏夹名称

​	`mid`:用户mid

​	`media_id`:收藏夹id

​	`fid`:作用未知，为收藏夹id前七位（十进制）

​	`cur_count`: 当前收藏夹中数量

​	`max_count`:最大数量

​	`cover`：最新三个收藏的列表

​		`aid`: 视频AV号

​		`pic`: 封面

​		`type`: 2

 

