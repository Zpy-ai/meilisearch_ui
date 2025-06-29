## 介绍
1.ai和nornal是hby配置的，ai为带有ai生成摘要和关键词。所用大模型为千问plus。  
2.mymeilisearch是我自己爬取艾瑞网报告信息上传到meilisearch容器，配置vector，实现语义搜索，semanticRatio=0，纯关键词搜索，无关键词会返回id较小的报告
## 结构
```plaintext
|--ai #ai生成摘要和关键词  
   | --images #界面图  
   |--ai.py  #代码  
   |--requirements  
|--normal    #普通版无ai  
   |--normal.py #代码  
   |--requirements  
|--test.py #最初版测试代码  
```
