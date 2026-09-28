---
title: "{{ replace .File.ContentBaseName "-" " " | title }}"
date: {{ .Date | time.Format "2006-01-02" }}
# lastmod: 2026-10-01   # set when you revise the note; shown as 更新 in the byline
description: ""         # one-line subtitle under the title
tags: []
draft: true
---

<!-- Page bundle: put images next to this index.md. Do not repeat the title as "# …".
Box:     > [!theorem] 定理 1（标题）   (every body line starts with "> ", blank line between boxes)
Figure:  {{ "{{</*" }} figure src="plot.svg" alt="…" caption="图注，不写“图 N：”" {{ "*/>}}" }}
Math:    $…$  \(…\)  $$…$$  \[…\] -->

## 1. 第一节

