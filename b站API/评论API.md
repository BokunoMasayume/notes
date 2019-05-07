评论API ：

> <https://api.bilibili.com/x/v2/reply?jsonp=jsonp&pn=1&type=1&oid=46105255&sort=0[&root=rpid]>

- sort: 排序方式 0/1
- pn: 分页
- type: 评论是否显示 1返回评论，2返回禁止评论信息
- oid: 视频AV号
- root: 楼层主评论id



- `code`
- `message`
- `ttl`
- `data`
  - `assist`
  - `blacklist`
  - `config`
  - `hots`
  - `mode`
  - `notice`
  - `page`: 数据库查询分页信息
    - `acount`: 总评论数
    - `count`: 总主评数（占楼层的）
    - `num`: 第几页
    - `size`: 分页大小
  - `replies`: 评论数据
    - `action`
    - `assist`
    - `attr`
    - `content`: 回复内容
      - `device`
      - `members`
      - `message`: 真正的内容
      - `plat`
    - `count`:  回复数
    - `rcount`: 好像也是回复数
    - `ctime`
    - `dialog`
    - `fansgrade`
    - `floor`:楼层
    - `folder`: 折叠？
    - `like`: 赞数
    - `member`: 评论者
      - `mid`
      - `uname`
      - `sign`
      - `sex`
      - `rank`
      - `pendant`
      - `avatar`: 头像
      - 。。。
    - `mid`
    - `oid`: AV号
    - `parent`: 回复的评论id
    - `parent_str`
    - `replies`: 部分楼中楼
      - `rpid`
      - `oid`
      - `content`: 内容
      - `count`: 回复数？
      - `dialog`:??
      - `floor`: 楼层
      - `like`
      - `mid`
      - `parent`
      - `root`
      - `member`
      - ...基本和上面一样
    - `root`
    - `root_str`
    - `rpid`
    - `rpid_str`
    - `state`
    - `type`: 是楼中楼还是啥啥
    - `up_action`
  - `support_mode`
  - `top`
  - `upper`???
  - `vote`











 

返回的json对象中，

`data`: 返回对象中最主要的成员对象

​	`hots`: 热评,comment对象数组

​		`page`: 

​		`num`:当前页数

​		`size`:分页评论数，固定值

​		`count`:似乎是视频总评论量

​		`acount`:似乎是带上楼中楼的总评论数

​	`upper`:

​		`mid`:up主id

​		`top`:up置顶评论

​	`replies`:本页评论(comment 对象)的数组



**comment对象**

​	`floor`: 楼层

​	`mid`: 用户id

​	`oid`:视频av号

​	`rpid`：该回复的id

​	`rpid_str`

​	`parent`：回复的评论id

​	`parent_str`

​	`rcount`:回复数

​	`root`: 回复的层主评论的id

​	`root_str`

​	`content`:

​		`message`:评论内容

​	`member`: 评论者用户信息

​		`avatar`:头像URL

​		`mid`:他id

​		`uname`: 昵称

​		`nameplate`:好像是勋章的显示用信息

​		`sex`:b站之前的迷之操作，随时挂着性别

​		`vip`:

​			`vipStatus`:1是现役大会员，0不是

​			`vipType`：看到过2[会员],1[非],0[非]，不知具体信息

