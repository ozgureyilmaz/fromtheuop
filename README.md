# Compilers for Machine Learning

A hands-on one semester course where students build their own compiler from scratch, starting from elementwise programs and ending with training SOTA LLMs on GPUs. This course aggressively builds on the previous week, and is an exercise in slop management. If you let any slop in early, it will compound and you will not finish the class.

Course description: This course covers the design and implementation of a modern machine learning compiler, and examines the interaction between IR design, hardware capabilities, and the structure of machine learning programs. Topics covered include term rewriting, code generation, movement operators, kernel fusion, memory hierarchies, GPU architecture, automatic differentiation, and flash attention. It is a project course, providing experience with performance-oriented programming, managing a codebase that grows all semester, and working in 1 or 2 person teams, culminating in a compiler capable of training modern LLMs.

Prerequisites: This course has no formal prerequisites, but it is designed for someone who is deeply familiar with the material in 15-213 or equivalent, highly skilled in the language they choose to implement their compiler in, capable of managing programming projects that are long in timeline, and comfortable with C and some GPU programming. There will be frequent in-person quizzes and lectures will not be posted online, so only sign up if you plan to attend class.

This syllabus will outline the spec, but not the implementation. The implementation and choice of programming language is up to the student. There is no starter code. We will establish a universal, simple, interchange format for UOps.

```
; uop v1 256x256 GEMM
%0 = buffer : dtype=f32 slot=0 size=65536
%1 = buffer : dtype=f32 slot=1 size=65536
%2 = reshape %0, (256, 256, 1) 
%3 = reshape %1, (256, 1, 256)
%4 = mul %2, %3
%5 = reduce %4 : op=add pop=1
```

---

## Week 1: UOps & Rewrites
**`uops` · `elementwise ops` · `symbolic` · `rewrites` · `renderer`**

Students become familiar with the `UOp` and write a compiler capable of compiling and simplifying elementwise programs. We introduce the graph structure and rewriting. The UOp is hash consed.

```
UOp(op, src:tuple[UOp, ...], arg)
```

Recursive properties introduced:
```
dtype(bool,i32,f32)
```

Ops introduced:
```
PARAM(ParamArg) -- arg input to the call
CALL(CallArg) -- src[0] is the function body, src[1:] are the params...
CONST -- type determined by the arg 
CAST/NEG/RECIP/EXP2/LOG2
ADD/MUL/CMPLT/CMPNE/MAX (note, for bool, MUL=AND, MAX=OR, CMPNE=XOR)
FLOORDIV/FLOORMOD
WHERE
```

The graph rewrite engine should be capable of "running" these programs via const folding.

## Week 2: Rendering & Runtime & Memory

Students write a renderer and a runtime capable of compiling and executing C programs.

Recursive properties introduced:
```
dtype(void,bool,i32,f32)
shape:tuple[int, ...]
addrspace(MEM,ALU)
```

Ops introduced:
```
BUFFER(ParamArg) -- global persistent buffer (never in call)
ALLOC(ParamArg) -- call scope storage, example "float a[10]"
INDEX -- shape from (n,) -> (), pop axes off the left with srcs
LOAD/STORE -- load has 1 src, store has 2
AFTER -- AFTER(buf, <store>)
GROUP -- group multiple void outputs together, unordered, shape is ()
```

This can render (loopless) C functions that act on memory and immediates.

## Week 3: Loops
**`loops` · `gemms` · `convs`**

Ops introduced:
```
RANGE -- src is the end of the range (3 means 0,1,2) and arg is the range name
END -- srcs are (passthrough, range)
```

Should be simple, and we can write gemms and convs in this language.

## Week 4: Movement Ops + Rangeify
**`movement ops` · `reduction` · `rangeify`**

Now we introduce movement ops and loops. Note how you can derive gemms and convs from the basic movement ops. 

Ops introduced:
```
RESHAPE/EXPAND/PERMUTE/FLIP/PAD/SHRINK
REDUCE (axis count / op)
```

Rangeify removes shapes from all ops.

## Week 5: Kernel splitting and fusion

This compiler is now capable of compiling *any* model, splitting it into kernels, and compiling it to C.

Ops introduced:
```
STAGE -- a temp buffer you can index into
```

## Week 6: Memory Hierarchies & Fast Kernels
**`memory hierarchies` · `upcasting` · `fast GEMMs` · `flash attention`**

Now things get fast. Still only on CPU, but we can now produce SOTA-competitive C code for GEMMs, reduces, and convs.

Many students will realize they have to rewrite their week 3-4.

Ops introduced:
```
STACK
```

## Weeks 7–8: Multithreading & GPUs & Hardware Accelerators
**`parallel` · `GPUs` · `hardware accelerators` · `tensor cores`**

Here we add kernels, GPUs, axis mappings, and tensor cores. This can produce decent torch-competitive CUDA/HIP code now.

## Weeks 9–10: Real Models & Autodiff
**`real models` · `autodiff`**

Here we implement a real LLM + autodiff to train models.

## Weeks 11–End: Student Projects

Use your compiler to implement a paper, port it to strange hardware, mostly anything

---

## At a Glance

| Weeks | Topics | Milestone |
|-------|--------|-----------|
| 1–2 | UOps, elementwise ops, symbolic, rewrites | Compile & simplify elementwise programs to C |
| 3–4 | Loops, movement ops, reduction, rangeify | Compile any model to C (slow, but correct) |
| 5–6 | Memory hierarchies, upcasting, GEMMs, convs | SOTA-competitive CPU code |
| 7–8 | Call, GPUs, accelerators, tensor cores | Torch-competitive CUDA/HIP code |
| 9–10 | Real models, autodiff | Train a real LLM |
| 11+ | Student projects | Choose a project and extend your compiler |
