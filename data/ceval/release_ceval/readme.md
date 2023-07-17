# C-Eval 数据集
 
Developed by TsingHua University, Shanghai Jiaotong University, and University of Edinburgh. 

Beta 测试版. Document by Yao Fu 

## 基础知识

一般有四种 prompting 范式，见 https://arxiv.org/abs/2301.12726 Figure 1B
* in-context answer-only <- 推荐从这里开始
* in-context chain-of-thought
* zero-shot answer-only
* zero-shot chain-of-thought

可用四种模式：
* in-context answer-only
* zero-shot answer-only
* in-context chain-of-thought
* zero-shot chain-of-thought

新手推荐从 in-context answer-only 开始 -- 我们最终测评的结果也会是 in-context answer-only

## 数据集

数据集一共有 50+ 科目，分为：
* `dev`: in-context few-shot demonstration 集合，每个科目有五个 in-context example 以及相对应的 explanation。推荐使用这里的 case 作为 in-context 的 prompt 
* `val`: validation 集合，一共 1000+ 题目，单科 20-30 题。推荐在这上面调超参数。这上面的单科效果不可信因为题目太少了，但是总的分数还是可信的
* `test`: 测试集，一共 15000+ 题目，单科 200 - 300 题，把模型在这上面预测，然后创建一个新的"模型预测"栏，把预测出来的答案填在，返回给符尧


## Prompt 的格式

如果模型还没有被调成一个 chatbot，那么：
```
以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。 <- 注意把 {subject} 改成具体的科目名称

[题目 1]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
让我们一步一步思考
[explanation]
答案：A

...

[题目 5]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
让我们一步一步思考
[explanation]
答案：C

[测试题目]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
让我们一步一步思考
[explanation] <- 模型从此处生成
答案：
```

上面是 in-context chain-of-thought 格式的 prompt。如果是 zero-shot 的话，则去掉 [题目 1] 到 [题目 5] 的 in-context 样本

如果是 answer-only 的话，则去掉 {让我们一步一步思考 [explanation]} 的内容

以下是 in-context answer-only prompt 的格式
 
```
以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。 <- 注意把 {subject} 改成具体的科目名称

[题目 1]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
答案：A

...

[题目 5]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
答案：C

[测试题目]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
答案：<- 模型从此处生成
```


如果模型已经调成一个 chatbot，那么 prompt 也需要改成对话的格式：
```
System: 
以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。

User:
[题目 1]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
让我们一步一步思考

Assistent:
[explanation]
答案：A

...

User:
[题目 5]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
让我们一步一步思考

Assistent:
[explanation]
答案：C

User:
[测试题目]
A: [选项 A 具体内容]
B: [选项 B 具体内容]
C: [选项 C 具体内容]
D: [选项 D 具体内容]
让我们一步一步思考

Assistent:
[explanation] <- 模型从此处生成
答案：
```

相应的，zero-shot 和 answer-only 版本的 prompt 需要分别去掉 in-context 样本和 {让我们一步一步思考 [explanation]} 的内容

Again，推荐使用 in-context answer-only 作为起点

更多关于 reasoning/ chain-of-thought 的内容，参见博客 

[Towards Complex Reasoning: the Polaris of Large Language Models](https://tinyurl.com/67c2eazt)

## Decoding 的方法

如果是 answer-only 模式
* 只看 A, B, C, D 的 logits，取最大的作为答案

如果是 chain-of-thought 模式
* 测试的时候一般 temperature 设置为 0 做 greedy decoding，因为这种 variance 低
* 大模型一般不用 beam search，贵且作用不大
* 上线一般用 sampling，因为用户友好，说错了可以再说一遍

## 理解 [推理] 和 [知识]

大模型测测试题目一般分推理和知识两种类型：
* 有些题目天生不需要 reasoning，比如中国语言文学里面一个是 “《茶馆》的作者是谁”，这种不需要 CoT，直接 AO 即可，CoT 反而增加了 distractor 
* 有些题目天生需要 reasoning，比如求定积分，这种直接给答案基本上都是随着直觉瞎猜，还是得一步一步推
* 一般而言，知识性的问题不大需要 CoT，推理型的问题需要 CoT 
* MMLU 是一个典型的知识型数据集，所以 PaLM 在这上面 AO 比 CoT 好
* BBH 是一个典型的推理型数据集，这上面 CoT 显著好于 AO

[知识] 和 [推理] 是两项可以显著区分大小模型的能力，其中
* [推理] 能力的区分度是最高的，比如说 gsm8k 这个数据集，GPT4 92 分，LLaMA 7b 只有七分，模型每大一点基本上都是十几二十分的差距
* [知识] 的区分度没有 [推理] 这么高，但也很高； 这里面模型每大一个台阶基本上是五六分的差距
* [推理] 能力小的模型基本没有，很多时候 acc 只有个位数
* [知识] 能力小模型也会有一点，比如 MMLU 上 11B flant5 也有 40+

## TODO
* [ ] 如何测试 CoT
* [ ] 如何解读结果
* [ ] 建立 `lib_prompt` 文件夹，把默认 prompt 搞成一个 .json 文件