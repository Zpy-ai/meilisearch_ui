## 介绍
ai和nornal是hby配置的，ai为带有ai生成摘要和关键词。所用大模型为千问plus。
mymeilisearch是我自己爬取艾瑞网报告信息所写的，
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
