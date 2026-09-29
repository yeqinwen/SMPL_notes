# Publish your Obsidian Notes

[SMPL_notes](https://yeqinwen.github.io/SMPL_notes/)，使用[MkDocs template](https://github.com/jobindjohn/obsidian-publish-mkdocs)构建，笔记网页保存在[github](https://github.com/yeqinwen/SMPL_notes)


1. obsidian：记笔记
2. github desktop：将笔记同步到github
3. github：保存笔记文件
4. MkDocs：将笔记发布为网页


# obsidian插件
1. Number Headings：自动标题
2. floating toc：浮动目录
3. code styler：方便保存、显示和复制代码


# 注意

1. 图片保存在每个目录下的 `assets`文件夹下；在添加图片时，需要在每个图片前面加上`assets/`，从而确保在obsidian、github和mkdoc网页上都能正常显示！
2. obsidian设置：文件与链接->链接-> 不勾选“使用wiki链接”(`![[assets/xxx.png]]` → `![](assets/xxx.png)`)；勾选“始终更新内部链接”
3. 在obsidian的插件Number Headings设置里面勾选skip top heading level，这样的话，obsidian里的层级标题与网页上的标题就是一致的。
4. 在将obsidian笔记同步到git时，可以使用两个工具，一个是github自带的github desktop，一个是obsidan的git插件。obsidan的git插件受网络影响比较大，github desktop同步要更好一些！

