# DS整理

## 绪论

### 数据的存储结构

#### 顺序映像

以相对的存储位置表示后继关系。

#### 链式映像

以附加信息（指针）表示后继关系

#### 抽象数据类型

可用（D，S，P）三元组表示

D：数据对象

S：D上的关系集

P：对D的基本操作集

### 算法

算法是为了解决某类问题而规定的一个有限长的操作序列。一个算法必须满足：有穷性、确定性、可行性、有输入、有输出。

#### 算法设计原则

正确性、可读性、健壮性、高效率与低存储量需求

#### 时间复杂度 

随着问题规模n的增长，算法执行时间的增长率和 f(n) 的增长率相同，则可记作：
$$
T(n) = O(f(n))
$$

#### 空间复杂度

若输入数据所占空间只取决于问题本身，和算法无关，则只需要分析除输入和程序之外的辅助变量所占额外空间。

若所需额外空间相对于输入数据量来说是常数，则称此算法为原地工作。

若所需存储量依赖于特定的输入，则通常按最坏情况考虑。

## 线性表

#### 定义



#### 顺序映像



#### 链式映像

##### 单链表

*问题*

1. 表长为一个隐含的值
2. 最后一个元素之后插入元素时，需遍历整个链表
3. 元素的 “位序” 概念淡化，结点的 “位置” 概念加强

*改进*

1. 增加 “表长”、“表尾指针”、“当前位置指针” 三个数据域
2. 将基本操作中的 “位序i” 改为 “指针p”

```c
typedef struct{
    Link head, tail;
    int len;      // 指示链表长度
    Link current; // 指向当前位置
} LinkList;
```

##### 双向链表

“插入” 和 “删除” 需要同时修改两个方向上的指针。

```c
// ListInsert(&L, i, e);
s->next = p->next;
p->next = s;
s->next->prior = s;
s->prior = p;
// ListDelete(&L, i);
p->next = p->next->next;
p->next->prior = p;
```



## 串

### 串的模式匹配算法

#### 简单算法

```c
int Index(SString S, SString T, int pos) {
    // 返回子串T在主串S中第pos个字符之后的位置。若不存在，
    // 则函数值为0。其中，T非空，1≤pos≤StrLength(S)。
    i = pos;   
    j = 1;
    while (i <= S[0] && j <= T[0]) {
        if (S[i] == T[j]) { 
            ++i;  
            ++j; 
        }   // 继续比较后继字符
        else { 
            i = i-j+2;   
            j = 1; 
        }     // 指针后退重新开始匹配
    }
    if (j > T[0])  
        return  i-T[0];
    else 
        return 0;
} // Index
```

算法时间复杂度：O(n×m)

#### 首尾匹配算法

先比较模式串的第一个字符，再比较模式串的最后一个字符，最后比较模式串中从第二个到第n-1个字符。

```c
int Index_FL(SString S, SString T, int pos) {
   sLength = S[0];  tLength = T[0];
   i = pos; 
   patStartChar = T[1];  patEndChar = T[tLength];
   while (i <= sLength – tLength + 1) {
     if (S[i] != patStartChar) ++i;  //重新查找匹配起始点
     else  if (S[i+tLength-1] != patEndChar) ++i; 
                                // 模式串的“尾字符”不匹配
     else {     // 检查中间字符的匹配情况
         k = 1;  j = 2;
         while ( j < tLength && S[i+k] == T[j])
             { ++k;   ++j; } 
         if ( j == tLength )  return i;
         else  ++i;   
             // 重新开始下一次的匹配检测
     }
   return 0;
  }
```

#### KMP算法

匹配过程中，主串的跟踪指针不回溯

算法时间效率达到：O(n＋m)

```c
int Index_KMP(SString S, SString T, int pos) {
     //  1≤pos≤StrLength(S)
     i = pos;  j = 1;
     while (i <= S[0] && j <= T[0]) {
         if (j = 0 || S[i] == T[j]) {
		 ++i;  ++j; }      // 继续比较后继字符
         else  j = next[j];         // j回溯，模式串右移
     }
    if (j > T[0])  return  i-T[0];    // 匹配成功
    else return 0;
} // Index_KMP
```

##### Next函数的定义

设next[j]=k，表示当模式串中第j个字符与主串中相应字符失配时，在模式串中需重新与主串中该字符进行比较的字符的位置。

算法时间复杂度：O(m)

求next函数值的过程是一个递推过程，分析如下:

```c
已知：next[1] = 0；
假设：next[j] = k1；检查 T[j] ?= T[k1]
若：    T[j] = T[k1]
则：    next[j+1] = next[j]+1=k1+1
若：    T[j] != T[k1]
则需往前回朔，
next[k1]=k2 ；检查 T[j] ?= T[k2]
若 T[j] ＝T[k2]
则next[j+1] = next[k2]+1=k2 +1
若：T[j] != T[k2]
则继续往前回朔，检查 T[j] ?= T[next[k2]]

void get_next(SString &T, int &next[] ) {
     // 求模式串T的next函数值并存入数组next
     i = 1;   next[1] = 0;   j = 0;
      while (i < T[0]) {
           if (j = 0 || T[i] == T[j])
                 {++i;  ++j; next[i] = j; }
           else  j = next[j];
      }
    } // get_next
```

##### 改进Next函数

例如：

​    S = ‘’aaabaaaaabaaabaaabaaab‘’

​    T = ‘’aaaab‘’

next[j]=01234

nextval[j]=00004

```c
void get_nextval(SString &T, int &nextval[]) {
      i = 1;   nextval[1] = 0;   j = 0;
      while (i < T[0]) {
          if (j = 0 || T[i] == T[j]) {
              ++i;  ++j;
              if (T[i] != T[j])  nextval[i] = j;
              else  nextval[i] = nextval[j];
         }
        else  j = nextval[j];
     }
  } // get_nextval
```

## 数组与广义表

### 矩阵的压缩存储

对特殊矩阵进行压缩存储时的下标变换公式

### 随机稀疏矩阵的压缩存储方法

#### 三元组顺序表

```c
#define  MAXSIZE  12500
 typedef struct {
     int  i, j;      //该非零元的行下标和列下标
     ElemType  e;    // 该非零元的值
 } Triple;  // 三元组类型
typedef struct {
     Triple  data[MAXSIZE]; 
      int       mu, nu, tu; 
} TSMatrix;  // 稀疏矩阵类型
```

##### 求转置矩阵

首先应该确定转置每一行的第一个非零元在三元组中的位置

时间复杂度为: O(M.nu+M.tu)

```c
Status FastTransposeSMatrix(TSMatrix M, TSMatrix &T){
    T.mu = M.nu;  T.nu = M.mu;  T.tu = M.tu;
    if (T.tu) {
        for (col=1; col<=M.nu; ++col)  
            num[col] = 0;
        for (t=1; t<=M.tu; ++t)  
            ++num[M.data[t].j];
        cpot[1] = 1;
        for (col=2; col<=M.nu; ++col)
            cpot[col] = cpot[col-1] + num[col-1];
        for (p=1; p<=M.tu; ++p) {                            
            col = M.data[p].j;
            q = cpot[col];
            T.data[q].i = M.data[p].j;
            T.data[q].j = M.data[p].i;
            T.data[q].e = M.data[p].e;
            ++cpot[col]
        }
    } // if
    return OK;
} // FastTransposeSMatrix
```

#### 行逻辑联结的顺序表

修改前述的稀疏矩阵的结构定义，增加一个数据成员rpos[MAXMN]，其值在稀疏矩阵的初始化函数中确定。

##### 矩阵乘法

```c
Q初始化；
  if  Q是非零矩阵 {  // 逐行求积
     for (arow=1; arow<=M.mu; ++arow) {
        //  处理M的每一行
        ctemp[] = 0;          // 累加器清零
       计算Q中第arow行的积并存入ctemp[] 中；
        将ctemp[] 中非零元压缩存储到Q.data；
     } // for arow
  } // if 
```

累加器ctemp初始化的时间复杂度为O(M.mu * N.nu)，

求Q的所有非零元的时间复杂度为O(M.tu * N.tu/N.mu)，

进行压缩存储的时间复杂度为O(M.mu * N.nu)，

总的时间复杂度就是O(M.mu * N.nu+M.tu * N.tu/N.mu)。

#### 十字链表

```c
// 节点定义：
typedef struct OLNode {
	int i, j;
	ElemType e;
	struct OLNode *right, *down;
}OLNode, *OLLink;
// 十字链表定义：
typedef struct CrossList {
	OLLink *rhead, *chead;
	int mu, nu, tu;
}
```

### 广义表

广义表 LS = ( a1, a2, …, an)的结构特点:

1. 广义表中的数据元素有相对次序；

2. 广义表的长度定义为最外层包含元素个数；

3. 广义表的深度定义为所含括弧的重数；

      注意：“原子”的深度为 0  

   ​           　“空表”的深度为 1 

4. 广义表可以共享；

5. 广义表可以是一个递归的表。

   递归表的深度是无穷值，长度是有限值。

6. 任何一个非空广义表    LS = ( a1, a2, …, an)

   均可分解为

   ​           表头  Head(LS) = a1   和

   ​           表尾  Tail(LS) = ( a2, …, an)     两部分。

##### 存储结构

通常采用头、尾指针的链表结构

1. 表头表尾分析法

```c
typedef enum {ATOM, LIST} ElemTag;
          // ATOM==0:原子, LIST==1:子表
typedef struct GLNode {
   ElemTag  tag;   // 标志域
   union{
     AtomType  atom;      // 原子结点的数据域
     struct {struct GLNode *hp, *tp;} ptr;
   };
} *GList
```

2. 子表分析法

##### 广义表操作的递归函数

- 分治法
  - 求广义表深度（子表深度+1）
  - 复制广义表（分别复制表头表尾）
  - 创建广义表（子表组合成广义表）
- 后置递归法
  - 删除广义表中所有元素为x的原子结点（删除时，不仅要删除原子结点，还需要删除相应的表结点）
- 回溯法

对于已求得的部分解 (x1, x2, …, xi) ，若在添加 xi+1之后仍然满足约束条件，得到一个新的部分解 (x1, x2, …, xi+1) , 继续添加 xi+2 并检查之。

若对于所有取值于集合Si+1的xi+1都不能得到新的满足约束条件的部分解(x1,x2,× × ×，xi+1 )，则从当前子组中删去xi, 回溯到前一个部分解(x1,x2, × × ×,xi-1 )，重新添加那些值集Si中尚未考察过的xi，看是否满足约束条件。

```c
void B(int i, int n) {
    // 假设已求得满足约束条件的部分解(x1,..., xi-1)，本函
    //数从 xi 起继续搜索，直到求得整个解(x1, x2, … xn)。
    if (i>n)
    else  
        while ( ! Empty(Si)) {
            从 Si 中取 xi 的一个值 vi;
            if (x1, x2, …, xi) 满足约束条件
                B( i+1, n);   // 继续求下一个部分解
            从 Si 中删除值 vi;
        }
} // B
```

**综合几点**

1. 对于含有递归特性的问题，最好设计递归形式的算法。
2. 实现递归函数，目前必须利用“栈”。
3. 分析递归算法的工具是递归树，从递归树上可以得到递归函数的各种相关信息。
4. 递归函数中的尾递归容易消除。
5. 可以用递归方程来表述递归函数的时间性能。

**Hanoi塔的递归函数**

```c
void hanoi (int n, char x, char y, char z)
{
    if (n==1)
        move(x, 1, z); 
    else {
        hanoi(n-1, x, z, y);  
        move(x, n, z);        
        hanoi(n-1, y, x, z);
    }
}   
```

## 树和二叉树

- 森林：是m（m≥0）棵互不相交的树的集合

- 满二叉树：指的是深度为k且含有2^k^-1个结点的二叉树。

- 完全二叉树：树中所含的 n 个结点和满二叉树中编号为 1 至 n 的结点一一对应。

### 性质

1. 在二叉树的第 i 层上至多有2^i-1^ 个结点。 (i≥1)
2. 深度为 k 的二叉树上至多含 2^k^-1个结点（k≥1）。
3. 对任何一棵二叉树，若它含有n0 个叶子结点、n2 个度为2 的结点，则必存在关系式：n0= n2+1。
4. 具有 n 个结点的完全二叉树的深度为 [log~2~n] +1 。
5. 若对含 n 个结点的完全二叉树从上到下且从左至右进行 1 至 n 的编号，则对完全二叉树中任意一个编号为 i 的结点： 
    (1) 若 i=1，则该结点是二叉树的根，无双亲，否则，编号为 [i/2] 的结点为其双亲结点；
    (2) 若 2i>n，则该结点无左孩子，   否则，编号为 2i 的结点为其左孩子结点；
    (3) 若 2i+1>n，则该结点无右孩子结点，否则，编号为2i+1 的结点为其右孩子结点。

### 顺序存储表示

双亲顺序表

```c
typedef struct BPTNode { // 结点结构
    TElemType  data;
    int  parent;     // 指向双亲的编号
    char  LRTag;    // 左、右孩子标志域
} BPTNode

typedef struct BPTree{ // 树结构
    BPTNode nodes[MAX_TREE_SIZE];
    int  num_node;     // 结点数目
    int  root;                // 根结点的位置
} BPTree
```

### 链式存储

#### 二叉链表

```c
typedef struct tagBiTNode { // 结点结构
    TElemType      data;
    struct tagBiTNode  *lchild, *rchild; // 左右孩子指针
} BiTNode, *BiTree;
```

#### 三叉链表

```c
typedef struct tagTriTNode { // 结点结构
      TElemType       data;
      struct tagTriTNode  *lchild, *rchild; // 左右孩子指针
      struct TriTNode  *parent;  //双亲指针 
   } TriTNode, *TriTree;
```

### 二叉树的遍历

#### 先（根）序的遍历算法

若二叉树为空树，则空操作；否则，

（1）访问根结点；

（2）先序遍历左子树；

（3）先序遍历右子树。

**算法的递归描述：**

```c
void Preorder (BiTree T, void( *visit)(TElemType& e))
{ // 先序遍历二叉树 
   if (T) {
      visit(T->data);            // 访问结点
      Preorder(T->lchild, visit); // 遍历左子树
      Preorder(T->rchild, visit);// 遍历右子树
   }
}
```

#### 中（根）序的遍历算法

**算法的非递归描述**

```c
void Inorder_I(BiTree T, void (*visit)(TelemType& e)){
    Stack *S;
    t = GoFarLeft(T, S);  // 找到最左下的结点
    while(t){
        visit(t->data);
        if (t->rchild)
            t = GoFarLeft(t->rchild, S);
        else if ( !StackEmpty(S ))    // 栈不空时退栈
            t = Pop(S);
        else    
            t = NULL; // 栈空表明遍历结束
    } // while
}// Inorder_I           
```

#### 后（根）序的遍历算法



#### 层次遍历

**算法的递归描述**：

```c
void  levelorder ( tree_ptr  tree )
{   
    enqueue ( tree );
    while (queue is not empty) {
        visit ( T = dequeue ( ) );
        for (each child C of T )
            enqueue ( C );
    }
}
```

### 遍历算法的应用举例

1. 查询二叉树中某个结点

2. 统计二叉树中叶子结点的个数

3. 求二叉树的深度(后序遍历)

   “访问结点”的操作：求得左、右子树深度的最大值，然后加1

4. 复制二叉树(后序遍历)

5. 建立二叉树的存储结构

   - 以字符串的形式建二叉树

   - 按给定的表达式建相应二叉树

   - 由二叉树的先序和中序序列建树

   - ```c
     void CrtBT(BiTree& T, char pre[], char ino[], int ps, int is, int n ) {
        // 已知pre[ps..ps+n-1]为二叉树的先序序列,ino[is..is+n-1]
         // 为二叉树的中序序列，本算法由此两个序列构造二叉链表  
        if (n==0) 
            T=NULL;
        else {
            k=Search(ino, pre[ps]); // 在中序序列中查询
            if (k== -1)  T=NULL;
            else { 
                T=(BiTNode*)malloc(sizeof(BiTNode));
     		   T->data = pre[ps];
     		   if (k==is)  
                    T->Lchild = NULL;
     		   else  
                    CrtBT(T->Lchild, pre[], ino[], ps+1, is, k-is );
                if (k=is+n-1) 
                    T->Rchild = NULL;
                else  
                    CrtBT(T->Rchild, pre[], ino[], ps+1+(k-is), k+1, n-(k-is)-1 );
            }
        } //
     } // CrtBT       
     ```

### 线索二叉树

遍历二叉树的结果是，求得结点的一个线性序列。指向该线性序列中的“前驱”和“后继” 的指针，称作“线索”。包含 “线索” 的存储结构，称作 “线索链表”。与其相应的二叉树，称作 “线索二叉树”。

#### **对线索链表中结点的约定**

在二叉链表的结点中增加两个标志域，并作如下规定：

- 若该结点的左子树不空，则Lchild域的指针指向其左子树，且左标志域的值为“指针 Link”；否则，Lchild域的指针指向其“前驱”，且左标志的值为“线索 Thread” 。（右子树同理）

```c
typedef enum { Link, Thread } PointerThr;  
     // Link=0:指针，Thread=1:线索
typedef struct BiThrNod {
   TElemType        data;
   struct BiThrNode  *lchild, *rchild;  // 左右指针
   PointerThr         LTag, RTag;    // 左右标志
} BiThrNode, *BiThrTree;
```

#### 线索链表的遍历算法

```c
for ( p = firstNode(T); p; p = Succ(p) )
      Visit (p);
```

#### 线索二叉树的遍历

中序遍历

```c
void InOrderTraverse_Thr(BiThrTree T, 
                                  void (*Visit)(TElemType e)) {
  p = T->lchild;       // p指向根结点
  while (p != T) {     // 空树或遍历结束时，p==T
     while (p->LTag==Link)  p = p->lchild;  // 第一个结点
     Visit(p->data);
     while (p->RTag==Thread && p->rchild!=T) {
         p = p->rchild;  Visit(p->data);      // 访问后继结点
     }
     p = p->rchild;          // p进至其右子树根
  }
} // InOrderTraverse_Thr
```

#### 线索二叉树的建立

在中序遍历过程中修改结点的左、右指针域，以保存当前访问结点的“前驱”和“后继”信息。遍历过程中，附设指针pre,  并始终保持指针pre指向当前访问的、指针p所指结点的前驱。

- 添加头结点

```c
   Thrt->LTag = Link;  Thrt->RTag =Thread; 
   Thrt->rchild = Thrt;      // 添加头结点
```

- 处理中间结点
- 处理最后一个结点

```c
pre->rchild = Thrt; 
pre->RTag = Thread;    
Thrt->rchild = pre;    
```

### 树和森林

#### 树的三种存储结构

1. 双亲表示法
2. 孩子链表表示法
3. 树的二叉链表（孩子-兄弟）存储表示法

#### 森林和二叉树的对应关系

设森林

​     F = ( T1, T2, …, Tn );

​     T1 = (root，t11, t12, …, t1m);

二叉树

​     B =( LBT, Node(root), RBT );

**由森林转换成二叉树的转换规则为:**

若 F = Φ，则 B = Φ；否则，

由 ROOT( T1 ) 对应得到 Node(root)；

由 (t11, t12, …, t1m ) 对应得到 LBT；

由 (T2, T3,…, Tn ) 对应得到 RBT。

**由二叉树转换为森林的转换规则为：**

若 B = Φ， 则 F = Φ；否则，

由 Node(root) 对应得到 ROOT( T1 )；

由LBT 对应得到 ( t11, t12, …，t1m)；

由RBT 对应得到 (T2, T3, …, Tn)。

#### 森林的遍历

##### 先序遍历

若森林不空，则访问森林中第一棵树的根结点；先序遍历森林中第一棵树的子树森林；先序遍历森林中(除第一棵树之外)其树构成的森林。

即依次从左至右对森林中的每一棵树进行**先根遍历**。

##### 中序遍历

若森林不空，则中序遍历森林中第一棵树的子树森林；访问森林中第一棵树的根结点；中序遍历森林中(除第一棵树之外)其余树构成的森林。

即依次从左至右对森林中的每一棵树进行**后根遍历**。

### 哈夫曼树与哈夫曼编码 

1. 根据给定的 n 个权值 {w1, w2, …, wn}，构造 n 棵二叉树的集合F = {T1,   T2,  … , Tn}，每棵二叉树中均只含一个带权值为wi 的根结点，其左、右子树为空树；
2. 在 F 中选取其根结点的权值为最小的两棵二叉树，分别作为左、右子树构造一棵新的二叉树，并置这棵新的二叉树根结点的权值为其左、右子树根结点的权值之和；
3. 从F中删去这两棵树，同时加入刚生成的新树；重复。

#### 前缀编码

任何一个元素的编码都不是同一元素集中另一个元素的编码的前缀。

利用赫夫曼树可以构造一种不等长的二进制编码，并且构造所得的赫夫曼编码是一种最优前缀编码，即使元素集合的编码序列总长度最短。

## 图

由一个顶点集V和一个弧集 VR构成的数据结构。

由于“弧”是有方向的，则称由顶点集和弧集构成的图为**有向图**。

由顶点集和边集构成的图称作**无向图**。

弧或边**带权**的图分别称作**有向网**或**无向网**。

假设图中有 n 个顶点，e条边，则

含有 e=n(n-1)/2 条边的无向图称作**完全图**；

含有 e=n(n-1) 条弧的有向图称作**有向完全图**；

假设一个连通图有 n 个顶点和e条边，其中n-1条边和n个顶点构成一个极小连通子图，称该极小连通子图为此连通图的**生成树**。

对非连通图，则称由各个连通分量的生成树的集合为此非连通图的**生成森林**。

对有向图，若只有一个顶点的入度为0,其余顶点的入度均为1，则是一棵有向树。

若有向图中含有若干棵不相交的有向树，这些有向树包含了所有的顶点，则这些树构成的森林就是**有向图的生成森林**。

### 图的存储表示

#### 图的数组(邻接矩阵)存储表示

```c
typedef struct ArcCell { // 弧的定义
     VRType  adj;    // VRType是顶点关系类型
             // 对无权图，用1或0表示相邻与否；
             // 对带权图，则为权值类型。
     InfoType  *info;  // 该弧相关信息的指针
} ArcCell,  AdjMatrix[MAX_VERTEX_NUM][MAX_VERTEX_NUM];

typedef struct { // 图的定义
     VertexType  vexs[MAX_VERTEX_NUM]; // 顶点信息
     AdjMatrix   arcs;      // 弧的信息                     
     int    vexnum, arcnum;   // 顶点数，弧数      
     GraphKind  kind;     // 图的种类标志             
} MGraph;
```

#### 图的邻接表存储表示

```c
typedef struct ArcNode {  
    int    adjvex;   // 该弧的狐头或狐尾的顶点所在的位置
    struct ArcNode  *nextarc; // 指向下一条弧的指针
    InfoType   *info;   // 该弧相关信息的指针
} ArcNode;

typedef struct VNode { 
    VertexType  data;   // 顶点信息
    ArcNode  *firstarc; // 指向第一条依附该顶点的弧
} VNode, AdjList[MAX_VERTEX_NUM];

typedef struct {  
    AdjList  vertices;
    int      vexnum, arcnum; 
    int      kind;          // 图的种类标志
} ALGraph;
```

#### 有向图的十字链表存储表示 

```c
typedef struct ArcBox { // 弧的结构表示
     int tailvex, headvex;   InfoType  *info;
     struct ArcBox  *hlink, *tlink;   
} VexNode;

typedef struct VexNode { // 顶点的结构表示
     VertexType  data;
     ArcBox  *firstin, *firstout;   
} VexNode;

typedef struct { 
   VexNode  xlist[MAX_VERTEX_NUM]; // 顶点结点(表头向量) 
   int   vexnum, arcnum; //有向图的当前顶点数和弧数
} OLGraph;
```

#### 无向图的邻接多重表存储表示

```c
typedef struct Ebox {
     VisitIf       mark;      // 访问标记
     int      ivex, jvex;   //该边依附的两个顶点的位置
     struct EBox  *ilink, *jlink; 
     InfoType     *info;          // 该边信息指针
} EBox;

typedef struct VexBox {
   VertexType  data;
   EBox  *firstedge; // 指向第一条依附该顶点的边
} VexBox;

typedef struct {  // 邻接多重表
    VexBox  adjmulist[MAX_VERTEX_NUM];
     int   vexnum, edgenum;    
} AMLGraph;
```

### 图的遍历

#### 深度优先搜索遍历图

从图中某个顶点V0 出发，访问此顶点，然后依次从V0的各个未被访问的邻接点出发深度优先搜索遍历图，直至图中所有和V0有路径相通的顶点都被访问到。

```c
void DFS(Graph G, int v) {
    // 从顶点v出发，深度优先搜索遍历连通图 G
    visited[v] = TRUE;   
    VisitFunc(v);
    for (w=FirstAdjVex(G, v); w!=0; w=NextAdjVex(G,v,w))
        if (!visited[w])  
            DFS(G, w); // 对v的尚未访问的邻接顶点 w 递归调用DFS
} // DFS
```

##### 非连通图的深度优先搜索遍历

首先将图中每个顶点的访问标志设为 FALSE,  之后搜索图中每个顶点，如果未被访问，则以该顶点为起始点，进行深度优先搜索遍历，否则继续检查下一顶点。

```c
void DFSTraverse(Graph G, Status (*Visit)(int v)) {
    VisitFunc = Visit;   
    for (v=0; v<G.vexnum; ++v) 
        visited[v] = FALSE; // 访问标志数组初始化
    for (v=0; v<G.vexnum; ++v) 
        if (!visited[v])  
            DFS(G, v); // 对尚未访问的顶点调用DFS
}
```

#### 广度优先搜索遍历图

从图中的某个顶点V0出发，并在访问此顶点之后依次访问V0的所有未被访问过的邻接点，之后按这些顶点被访问的先后次序依次访问它们的邻接点，直至图中所有和V0有路径相通的顶点都被访问到。

若此时图中尚有顶点未被访问，则另选图中一个未曾被访问的顶点作起始点，重复上述过程，直至图中所有顶点都被访问到为止。

```c
void BFSTraverse(Graph G, Status (*Visit)(int v)){
    for (v=0; v<G.vexnum; ++v)
        visited[v] = FALSE;  //初始化访问标志
    InitQueue(Q);       // 置空的辅助队列Q
    for ( v=0; v<G.vexnum; ++v )
        if ( !visited[v]) {    // v 尚未访问
            visited[v] = TRUE;  
            Visit(v);    // 访问v
            EnQueue(Q, v);     // v入队列
            while (!QueueEmpty(Q)) {
   		        DeQueue(Q, u); // 队头元素出队并置为u
                for(w=FirstAdjVex(G, u); w!=0;w=NextAdjVex(G,u,w))
                    if (!visited[w]) {
                        visited[w]=TRUE;  
                        Visit(w);
                        EnQueue(Q, w); // 访问的顶点w入队列
                    } // if
             } // while
         } // if 
} // BFSTraverse
```

##### 求两个顶点之间的一条路径长度最短的路径

### (连通网的)最小生成树

构造网的一棵最小生成树，即：在 e 条带权的边中选取 n-1 条边（不构成回路），使“权值之和”为最小。

#### 普里姆算法

取图中任意一个顶点v作为生成树的根，之后往生成树上添加新的顶点w。在添加的顶点 w 和已经在生成树上的顶点v 之间必定存在一条边，并且该边的权值在所有连通顶点 v 和 w 之间的边中取值最小。之后继续往生成树上添加顶点，直至生成树上含有 n个顶点为止。

应在所有连通U中顶点和V-U中顶点的边中选取权值最小的边。

```c
struct {
    VertexType  adjvex;  // U集中的顶点序号
    VRType     lowcost;  // 边的权值
} closedge[MAX_VERTEX_NUM];

void MiniSpanTree_P(MGraph G, VertexType u) {
    //用普里姆算法从顶点u出发构造网G的最小生成树
    k = LocateVex ( G, u ); 
    for ( j=0; j<G.vexnum; ++j )  // 辅助数组初始化
        if (j!=k)  
            closedge[j] = { u, G.arcs[k][j].adj };  
    closedge[k].lowcost = 0;      // 初始，U＝{u}
    for (i=1; i<G.vexnum; ++i) {
        // 继续向生成树上添加顶点;
        k = minimum(closedge);  // 求出加入生成树的下一个顶点(k)
        printf(closedge[k].adjvex, G.vexs[k]); // 输出生成树上一条边
  		closedge[k].lowcost = 0;    // 第k顶点并入U集
  		for (j=0; j<G.vexnum; ++j)  //修改其它顶点的最小边
    	    if (G.arcs[k][j].adj < closedge[j].lowcost)
      	        closedge[j] = { G.vexs[k], G.arcs[k][j].adj }; 
    }
}
```

时间复杂度：    O(n+(n-1)((n-1) + n))

​       			= O((n-1)2+n+n(n-1))

​			= O((n-1)2 +n2) = O(n^2^)

#### 克鲁斯卡尔算法

先构造一个只含 n 个顶点的子图SG，然后从权值最小的边开始，若它的添加不使SG中产生回路，则在SG上加上这条边，如此重复，直至加上n-1条边为止。

```c
构造非连通图 ST=( V,{ } );
k = i = 0;    // k 计选中的边数
while (k<n-1) {
 	++i;
    检查边集 E 中第 i 条权值最小的边(u,v);
  	若(u,v)加入ST后不使ST中产生回路，
  	则  输出边(u,v);  且  k++;
}
```

时间复杂度：O(eloge)

### 重（双）连通图和关节点

若从一个连通图中删去任何一个顶点及其相关联的边，它仍为一个连通图的话，则该连通图被称为重（双）连通图。

若连通图中的某个顶点和其相关联的边被删去之后，该连通图被分割成两个或两个以上的连通分量，则称此顶点为关节点。

**判别**：没有关节点的连通图为重连通图。

**判别：**

假设从某个顶点V0出发对连通图进行深度优先搜索遍历，则可得到一棵深度优先生成树，树上包含图的所有顶点。

若生成树的**根结点**，有两个或两个以上的分支，则此顶点(生成树的根)必为关节点；

对生成树上的任意一个“顶点”，若其某棵子树的根或子树中的其它“顶点”没有和其祖先相通的回边，则该“顶点”必为关节点。 

```c
void FindArticul(ALGraph G) {
    //设V0为深度优先遍历的出发点
    count = 1; 
    visited[0] = 1;
    for (i = 1; i < G.vexnum; ++i) 
        visited[i] = 0;
    p = G.vertices[0].firstarc;   
    v = p->adjvex;
    DFSArticul(G, v);    // 从第v顶点出发深度优先搜索
    if (count < G.vexnum) {  // 生成树的根有至少两棵子树
        printf (0, G.vertices[0].data);   // 根是关节点
        while (p->nextarc) {  // 下一棵子树
            p = p->nextarc; 
            v = p->adjvex;
            if (visited[v] == 0)
                DFSArticul(G, v);
        }
    ｝
}
```

- 定义函数: `low(v) = Min{visited[v], low[w], visited[k] }`

其中: 顶点w 是生成树上顶点v 的孩子；

​          顶点k 是生成树上和顶点v由回边相联接的祖先；

​          visited记录深度优先遍历时的访问次序。

若对顶点v，在生成树上存在一个子树根w，且 `low[w]  ≥ visited[v]`则顶点v为关节点。

- 对深度优先遍历算法作如下修改：
  - visited[v]的值改为遍历过程中顶点的访问次序count值；
  - 遍历过程中求得`low[v]=Min{visited[v],low[w],visited[k]}`
  - 从子树遍历返回时，判别`low[w]≥visited[v]?`

```c
void  DFSArticul(ALGraph G, int v0) {
    // 从第v0个顶点出发深度优先遍历图 G,
    // 查找并输出关节点
	min =visited[v0] = ++count;  
	// v0是第count个访问的顶点, 并设low[v0]的初值为min
	for(p=G.vertices[v0].firstarc; p; p=p->nextarc) {   
  	    // 检查v0的每个邻接点
        w = p->adjvex;       // w为v0的邻接顶点
    	if (visited[w] == 0) {  // w未曾被访问
    		DFSArticul(G, w);  // 返回前求得low[w]
        	if (low[w] < min)   
                min = low[w]; 
          	if (low[w]>=visited[v0])
   			    printf(v0, G.vertices[v0].data); //输出关节点
 		 }                       
 		 else     // w是回边上的顶点
             if (visited[w] < min)   
                 min = visited[w]; 
	}
	low[v0] = min;
} // DFSArticul
```

### 两点之间的最短路径问题

#### 迪杰斯特拉算法

求从某个源点到其余各点的最短路径

设置辅助数组Dist，其中每个分量Dist[k] 表示当前所求得的从源点到其余各顶点 k 的最短路径。

一般情况下，

Dist[k] = <源点到顶点 k 的弧上的权值>

或者 = <源点到其它顶点的路径长度> + <其它顶点到顶点 k 的弧上的权值>

1. 在所有从源点出发的弧中选取一条权值最小的弧，即为第一条最短路径。

2. 修改其它各顶点的Dist[k]值。假设求得最短路径的顶点为u，

   若 `Dist[u]+G.arcs[u][k]<Dist[k]`

   则将 `Dist[k]` 改为 `Dist[u]+G.arcs[u][k]`。

```c
void ShortestPath_DIJ (MGraph G,int v0,int P[][MAX_VERTEX_NUM],int D[] ) {
    // 求有向网G的v0顶点到其余顶点v的最短路径P[v]及其带权路径长度D[v]
    // 若P[v][0]≠0,表明从源点出发存在一条到顶点v的最短路径，该路径存放在P[v]中
    // final[v]为True则表明已经找到从v0到v的最短路径   
    for (v=1;v<=G.vexnum;v++)  { //初始化   
        final[v]=FALSE; 
        D[v]=G.arcs[v0][v];
        for(i=0;i<=G.vexnum;i++) 
            P[v]=0; //设空路径    	
        if(D[v]<INFINITY) 
            P[v][0]=v0; //若从v0到v有直达路径
    }
    D[v0]=0; 
    final[v0]=True; //初始时，v0属于S集
    //主循环，每次求得v0到某个顶点v的最短路径，并加v到S集  
    for (i=1;i<=G.vexnum;i++)  {//寻找其余G.vexnum-1个顶点
    	v=0;
        min=INFINITY;
        for(w=1;w<=G.vexnum;w++) {
            //寻找当前离v0最近的顶点v
            if((!final[w])&&(D[w]<min)) {	
                v=w; 
                min=D[w];
            }
        }
        if(!v) //若v=0表明所有与v0有通路的顶点均已找到了最短路径，退出主循环
            break;  
        final[v] = TRUE; //将v加入S集
        for(j=0;P[v][j]!=0;j++);
        P[v][j]=v;     //将路径P[v]延伸到顶点v
        for(w=1;w<=G.vexnum;w++) { //更新当前最短路径及距离
            if(!final[w]&&(min+G.arcs[v][w]<D[w])) {
                D[w]=min+G.arcs[v][w];
                for(j=0;P[v][j]!=0;j++) {
                    P[w][j] = P[v][j];
                }
            }
        }
    }
} // ShortestPath_DIJ

```

#### 弗洛伊德算法

从 vi 到 vj 的所有可能存在的路径中，选出一条长度最短的路径。

```c
若<vi,vj>存在，则存在路径{vi,vj}
                 // 路径中不含其它顶点
若<vi,v1>,<v1,vj>存在，则存在路径{vi,v1,vj}
             // 路径中所含顶点序号不大于1
若{vi,…,v2}, {v2,…,vj}存在，则存在一条路径{vi, …, v2, …vj}
             // 路径中所含顶点序号不大于2
...
```

依次类推，则 vi 至vj的最短路径应是上述这些路径中，路径长度最小者。

### 拓扑排序

检查有向图中是否存在回路的方法之一，是对有向图进行拓扑排序。

按照有向图给出的次序关系，将图中顶点排成一个线性序列，对于有向图中没有限定次序关系的顶点，则可以人为加上任意的次序关系。由此所得顶点的线性序列称之为拓扑有序序列。

对于有回路的有向图，不能求得它的拓扑有序序列。

#### 如何进行拓扑排序

1. 从有向图中选取一个没有前驱的顶点（入度为零的顶点），并输出之；
2. 从有向图中删去此顶点以及所有以它为尾的弧（弧头顶点的入度减1）；
3. 重复上述两步，直至图空，或者图不空但找不到无前驱的顶点为止。

```c
取入度为零的顶点v;
while (v<>0) {
     printf(v);  ++m;
     w:=FirstAdj(v);
     while (w<>0) {
        inDegree[w]--;
        w:=nextAdj(v,w);
     }
     取下一个入度为零的顶点v;
}
if m<n  printf(“图中有回路”);
```

为避免每次都要搜索入度为零的顶点，在算法中设置一个“栈”，以保存“入度为零”的顶点。

```c
CountInDegree(G,indegree);
        //对各顶点求入度
InitStack(S);
for ( i=0; i<G.vexnum; ++i)
   if (!indegree[i])  Push(S, i);
       //入度为零的顶点入栈
```

```c
count=0;           //对输出顶点计数
while (!EmptyStack(S)) {
  Pop(S, v);  
  ++count;
  printf(v);
  for (w=FirstAdj(v); w;  w=NextAdj(G,v,w)){
     --indegree(w);  // 弧头顶点的入度减一
     if (!indegree[w])  
         Push(S, w);  //新产生的入度为零的顶点入栈  
  }
}
if (count<G.vexnum) printf(“图中有回路”)
```

### 关键路径

整个工程完成的时间为：从有向图的源点到汇点的最长路径。

“**关键活动**”指的是：该弧上的权值增加 将使有向图上的最长路径的长度增加。

“**事件(顶点)**” 的 **最早**发生时间 `ve(j)` = 从源点到顶点j的最长路径长度；

“事件(顶点)” 的 **最迟**发生时间 `vl(k)` 取决于从顶点k到汇点的最长路径长度。

 假设第 i 条弧为 <j, k>， 则对第 i 项活动言

  “**活动(弧)**”的 **最早**开始时间 `ee(i) = ve(j)`；

  “活动(弧)”的 **最迟**开始时间 `el(i) = vl(j) = vl(k) – dut(<j,k>)`；

若`ee(i) == el(i)`，则第i条弧即为关键活动  `ve(j) == vl(j)`

####  事件发生时间的计算公式

```c
ve(源点) = 0；
ve(k) = Max{ve(j) + dut(<j, k>)}

vl(汇点) = ve(汇点)；
vl(j) = Min{vl(k) – dut(<j, k>)}
```

- 求ve的顺序应该是按拓扑有序的次序；

- 求vl的顺序应该是按拓扑逆序的次序；

拓扑逆序序列即为拓扑有序序列的逆序列，因此应该在拓扑排序的过程中，另设一个“栈”记下拓扑有序序列。

```c
void ToplogicalOrder (ALGraph G) {
    // 有向网G采用邻接表存储结构，求各顶点的最早发生时间ve
    // T为拓扑逆序顶点栈，S为零入度顶点栈
    // 若G无回路，则用栈T返回G的一个拓扑逆序
    FindInDegree(G, indegree)； // 对各顶点求入度indegree[]
    InitStack(T); 
    count = 0; 
    ve[0..G.vexnum - 1] = 0; // 初始化
    while(!StackEmpty(S))  { 
         Pop(S, j); Push(T, j); ++count;
        for(p = g.vertices[j].firstarc; p; p = p->nextarc) {
              k = p->adjvex;
              // 对j号顶点的每个邻接点的入度减1
              // 若入度减为0,则入栈
              if (--indegree[k] == 0) 
                  Push(S,k);
              if (ve[j] + *(p->info) > 
                  ve[k]) ve[k] = ve[j] + *(p->info);
    } // while
    if (count < G.vexnum)  
        return ERROR;
    else 
        return OK;
} // ToplogicalOrder

void CriticalPath (ALGraph G) {
    // G为有向网，输出G的各类关键活动
    if (!TopologicalOrder(G)) return ERROR;
    vl[0..G.vexnum - 1] = ve[g.vexnum -1]; //初始化事件的vl值
    while(!StackEmpty(T))  { // 按拓扑逆序求各顶点的vl值
        for (Pop(T, j), p = G.vertices[j].firstarc; p; 
                                    p = p->nextarc) {
            k = p->adjvex; 
            dut = *(p->info); // dut<j, k>
            if (vl[k] - dut < vl[j]) 
                vl[j] = vl[k] － dut;
         } // for 
    } // while
    for (j = 0; j < G.vexnum; j++) { // 求ee、el和关键活动
          for (p = G.vertices[j].firstarc; p; p = p->nextarc) {
                k = p->adjvex; dut = *(p->info);
                ee = ve[j]; el = vl[k] – dut;
                tag = (ee == el) ? ‘*’ : ‘ ’;
 	            printf(j, k, dut, ee, el, tag); // 输出关键活动
          } // for 
    } // for
 } // CriticalPath
```

## 排序

内部排序：若整个排序过程不需要访问外存便能完成

外部排序：若参加排序的记录数量很大，整个序列的排序过程不可能在内存中完成

### 插入排序

1. 在R[1..i-1]中查找R[i]的插入位置，R[1..j].key <= R[i].key <= R[j+1..i-1].key；

2. 将R[j+1..i-1]中的所有记录均后移一个位置；
3. 将R[i] 插入(复制)到R[j+1]的位置上。

#### 直接插入排序

顺序查找

```c
void InsertionSort ( SqList &L ) {
  // 对顺序表 L 作直接插入排序。
   for ( i=2; i<=L.length; ++i ) 
       if (L.r[i].key < L.r[i-1].key) {
			L.r[0] = L.r[i];            // 复制为监视哨
			for ( j=i-1; L.r[0].key < L.r[j].key;  -- j )
    			L.r[j+1] = L.r[j];        // 记录后移
			L.r[j+1] = L.r[0];        // 插入到正确位置
        }
} // InsertSort
```

**时间分析**

最好的情况（关键字在记录序列中顺序有序）：

“比较”的次数：$\sum_{i=2}^n1=n-1$

“移动”的次数：0

最坏的情况（关键字在记录序列中逆序有序）：

“比较”的次数：$\sum_{i=2}^ni=\frac{(n+2)(n-1)}{2}$

“移动”的次数：$\sum_{i=2}^n(i+1)=\frac{(n+4)(n-1)}{2}$

#### 折半插入排序

折半查找

```c
void BiInsertionSort ( SqList &L ) {
    for ( i=2; i<=L.length; ++i ) {
        L.r[0] = L.r[i];      // 将 L.r[i] 暂存到 L.r[0]
        // 在 L.r[1..i-1]中折半查找插入位置；
        low = 1;   
        high = i-1;
        while (low<=high) { 
	        m = (low+high)/2;           // 折半
	        if (L.r[0].key < L.r[m].key)
                high = m-1;   // 插入点在低半区
	        else  low = m+1; // 插入点在高半区
        }
        for ( j=i-1;  j>=high+1;  --j )
            L.r[j+1] = L.r[j];      // 记录后移
        L.r[high+1] = L.r[0];  // 插入
    } // for
} // BInsertSort
```

#### 表插入排序

利用静态链表进行排序，并在排序完成之后，一次性地调整各个记录相互之间的位置

首先将链表中数组下标为 1 的结点和表头结点构成一个循环链表，然后将后序的所有结点按照其存储的关键字的大小，依次插入到循环链表中。

```c
void LInsertionSort (Elem SL[ ]， int n){
    // 对记录序列SL[1..n]作表插入排序
    SL[0].key = MAXINT ;
    SL[0].next = 1;  
    SL[1].next = 0;
    for ( i=2; i<=n; ++i ){
        for ( j=0, k = SL[0].next; SL[k].key<=SL[i].key;
              j=k, k=SL[j].next );
        // 结点i插入在结点j和结点k之间
        SL[j].next = i;  
        SL[i].next = k;
    }
}// LinsertionSort
```

在排序之后调整记录序列，算法中使用了三个指针：

其中：p指示第i个记录的当前位置

​            i指示第i个记录应在的位置

​            q指示第i+1个记录的当前位置

```c
void Arrange ( Elem SL[ ], int n ) {
    p = SL[0].next;   // p指示第一个记录的当前位置
    for ( i=1; i<n; ++i ) {      
        while (p<i)  // p可能会指向已经排序过的位置，此时要找被移走的记录
            p = SL[p].next;
        q = SL[p].next; // q指示尚未调整的表尾
        if ( p!= i ) {
            SL[p]←→SL[i]; // 交换记录，使第i个记录到位
            SL[i].next = p;    // 指向被移走的记录，以后可以找回来
        }
        p = q; // p指示尚未调整的表尾，为找第i+1个记录作准备
    }
} // Arrange
```

#### 希尔排序(缩小增量排序)

对待排记录序列先作“宏观”调整，再作“微观”调整。所谓“宏观”调整，指的是，“跳跃式”的插入排序。

将记录序列分成若干子序列，分别对每个子序列进行插入排序。

例如：将 n 个记录分成 d 个子序列：

  { R[1]，R[1+d]，R[1+2d]，…，R[1+kd] }

  { R[2]，R[2+d]，R[2+2d]，…，R[2+kd] }

​    …

  { R[d]，R[2d]，R[3d]，…，R[kd]，R[(k+1)d] }

其中，d 称为增量，它的值在排序过程中从大到小逐渐缩小，直至最后一趟排序减为 1。

```c
void ShellInsert ( SqList &L, int dk ) {
    for ( i=dk+1; i<=n; ++i )
        if ( L.r[i].key< L.r[i-dk].key) {
            L.r[0] = L.r[i];     // 暂存在R[0]
            for (j=i-dk; j>0&&(L.r[0].key<L.r[j].key);
                 j-=dk)
                L.r[j+dk] = L.r[j];  
                // 记录后移，查找插入位置
            L.r[j+dk] = L.r[0];  // 插入
         } // if
} // ShellInsert

void ShellSort (SqList &L, int dlta[], int t)
{   // 增量为dlta[]的希尔排序
    for (k=0; k<t; ++t)
        ShellInsert(L, dlta[k]); 
        //一趟增量为dlta[k]的插入排序
} // ShellSort
```

### 快速排序

#### 起泡排序

比较相邻记录，将关键字最大的记录交换到n-i+1 的位置上

1. 起泡排序的结束条件为，最后一趟没有进行“交换记录”。
2. 一般情况下，每经过一趟“起泡”，“i 减一”，但并不是每趟都如此。

```c
void BubbleSort(Elem R[ ], int n) {
   i = n;
   while (i>1) { 
       lastExchangeIndex = 1;
       for (j = 1; j < i; j++)
           if (R[j+1].key < R[j].key) { 
       	      Swap(R[j], R[j+1]);
              lastExchangeIndex = j;  //记下进行交换的记录位置
           } //if
       i = lastExchangeIndex; // 本趟进行过交换的最后一个记录的位置
   } // while
} // BubbleSort
```

**时间分析**

最好的情况（关键字在记录序列中顺序有序）：

“比较”的次数：$n-1$

“移动”的次数：0

最坏的情况（关键字在记录序列中逆序有序）：

“比较”的次数：$\sum_{i=n}^2(i-1)=\frac{n(n-1)}{2}$

“移动”的次数：$3\sum_{i=n}^2(i-1)=\frac{3n(n-1)}{2}$

#### 一趟快速排序（一次划分）

目标：找一个记录，以它的关键字作为“枢轴”，凡其关键字小于枢轴的记录均移动至该记录之前，反之，凡关键字大于枢轴的记录均移动至该记录之后。

致使一趟排序之后，记录的无序序列R[s..t]将分割成两部分：R[s..i-1]和R[i+1..t]，且  

​       R[j].key≤ R[i].key ≤ R[j].key

​      (s≤j≤i-1)    枢轴     (i+1≤j≤t)

1）设置两个变量i、j，排序开始的时候：i=0，j=N-1；
2）以第一个数组元素作为关键数据，赋值给key，即 key=A[0]；
3）从j开始向前搜索，即由后开始向前搜索（j -- ），找到第一个小于key的值A[j]，A[i]与A[j]交换；
4）从i开始向后搜索，即由前开始向后搜索（i ++ ），找到第一个大于key的A[i]，A[i]与A[j]交换；
5）重复第3、4、5步，直到 I=J； (3,4步是在程序中没找到时候j=j-1，i=i+1，直至找到为止。找到并交换的时候i， j指针位置不变。另外当i=j这过程一定正好是i+或j-完成的最后令循环结束。）

```c
int Partition (RcdType R[], int low, int high) {
    R[0] = R[low];  
    pivotkey = R[low].key;  // 枢轴
    while (low<high) {
        while(low<high&& R[high].key>=pivotkey)
            -- high;      // 从右向左搜索
        R[low] = R[high];
        while (low<high && R[low].key<=pivotkey) 
            ++ low;      // 从左向右搜索
        R[high] = R[low];
    }
    R[low] = R[0];     
    return low; // 返回主元位置
}// Partition       
```

#### 快速排序

首先对无序的记录序列进行“一次划分”，之后分别对分割所得两个子序列“递归”进行快速排序。

```c
void QSort (RcdType & R[],  int s,  int  t ) {
    // 对记录序列R[s..t]进行快速排序
    if (s < t) {   // 长度大于1
        pivotloc = Partition(R, s, t); // 对 R[s..t] 进行一次划分
        QSort(R, s, pivotloc-1);
        // 对低子序列递归排序，pivotloc是枢轴位置
        QSort(R, pivotloc+1, t); //对高子序列递归排序
    }
} // QSort

void QuickSort( SqList & L) {
    // 对顺序表进行快速排序
    QSort(L.r, 1, L.length);
} // QuickSort
```

**时间分析**

假设一次划分所得枢轴位置 i=k，则对n 个记录进行快排所需时间：

T(n)= T~pass~(n) + T(k-1) + T(n-k)

其中Tpass(n)为对n 个记录进行一次划分所需时间。

若待排序列中记录的关键字是随机分布的，则k 取 1 至 n 中任意一值的可能性相同。

由此可得快速排序所需时间的平均值为：
$$
T_{avg}(n)={\rm c}n+\frac1 n\sum_{k=1}^n[T_{avg}(k-1)+T_{avg}(n-k)]
$$
设$T_{avg}<=b$

可得结果
$$
T_{avg}(n)<(\frac b 2+2c)(n+1){\rm ln}(n+1)
$$
结论: 快速排序的时间复杂度为O(nlogn)

若待排记录的初始状态为按关键字有序时，快速排序将蜕化为起泡排序，其时间复杂度为O(n^2^)。

为避免出现这种情况，需在进行一次划分之前，进行“予处理”，即：

先对 R(s).key, R(t).key 和R[\[(s+t)/2]].key，进行相互比较，然后取关键字为“三者之中”的记录为枢轴记录。

### 堆排序

#### 简单选择排序

```c
void SelectSort (Elem R[], int n ) {
    for (i=1; i<n; ++i) {
        // 选择第 i 小的记录，并交换到位
        j = SelectMinKey(R, i);       
        // 在 R[i..n] 中选择关键字最小的记录
        if (i!=j)  
            R[i]←→R[j]; // 与第 i 个记录交换
    }
} // SelectSort
```

**时间分析**

关键字间的比较次数总计为：$\sum_{i=1}^{n-1}(n-i)=\frac{n(n-1)}2$

移动记录的次数，最小值为 0, 最大值为3(n-1) 。

#### 堆排序

堆是满足下列性质的数列{r1, r2, …，rn}：

小顶堆：$\begin{cases} r_i\le r_{2i} \\ r_i\le r_{2i+1}\end{cases}$，用于降序排列

大顶堆：$\begin{cases} r_i\ge r_{2i} \\ r_i\ge r_{2i+1}\end{cases}$，用于升序排列

若将该数列视作完全二叉树，则 r~2i~ 是 ri 的左孩子； r~2i+1~ 是 ri 的右孩子。

堆排序即是利用堆的特性对记录序列进行排序的一种排序方法。

1. 创建一个堆 H[0……n-1]；
2. 把堆首（最大值）和堆尾互换；
3. 把堆的尺寸缩小 1，并调用 shift_down(0)，目的是把新的数组顶端数据调整到相应位置；
4. 重复步骤 2，直到堆的尺寸为 1。

```c
void HeapSort ( HeapType &H ) {
    // 对顺序表 H 进行堆排序
    for ( i=H.length/2; i>0; --i )
        HeapAdjust ( H.r, i, H.length ); 
        // 建大顶堆
    for ( i=H.length; i>1; --i ) {
        H.r[1]←→H.r[i];           
        // 将堆顶记录和当前未经排序子序列H.r[1..i]中最后一个记录相互交换
        HeapAdjust(H.r, 1, i-1); 
        // 对 H.r[1] 进行筛选
    }
} // HeapSort
```

所谓“筛选”指的是，对一棵左/右子树均为堆的完全二叉树，“调整”根结点使整个二叉树也成为一个堆。

```c
void HeapAdjust (RcdType &R[], int s, int m){   
    // 已知 R[s..m]中记录的关键字除 R[s] 之外均满足堆的特征
    rc = R[s];         // 暂存 R[s] 
    for ( j=2*s; j<=m; j *= 2 ) { // j 初值指向左孩子
        if ( j<m && R[j].key<R[j+1].key )  
            ++j;       // 令 j 指示左/右“子树根”较大的位置     
        if ( rc.key >= R[j].key )  
            break;     // 说明已找到 rc 的插入位置 s ，不需要继续往下调整
        R[s] = R[j];   // 否则记录上移，尚需继续往下调整
        s = j;    
    }
    R[s] = rc;         // 将调整前的堆顶记录插入到 s 位置
} // HeapAdjust
```

建堆是一个从下往上进行“筛选”的过程。

分别“筛选”子树，最后调整根结点。

**时间分析**

1. 对深度为 k 的堆，“筛选”所需进行的关键字比较的次数至多为2(k-1)；

2. 对 n个关键字，建成深度为h(=[log~2~n]+1)的堆，所需进行的关键字比较的次数至多 4n；

3. 调整“堆顶” n-1 次，总共进行的关键字比较的次数不超过

    2 ([log~2~(n-1)]+ [log~2~(n-2)]+ …+log~2~2) < 2n([log~2~n]) 

因此，堆排序的时间复杂度为O(nlogn)。

### 归并排序

将两个或两个以上的有序子序列 “归并” 为一个有序序列。

在内部排序中，通常采用的是2-路归并排序。即：将两个位置相邻的记录有序子序列归并为一个记录的有序序列。

```c
void Merge (RcdType SR[], RcdType &TR[], 
            int i, int m, int n) {
  // 将有序的记录序列 SR[i..m] 和 SR[m+1..n]
  // 归并为有序的记录序列 TR[i..n]
    for (j=m+1, k=i;  i<=m && j<=n;  ++k) {  
        // 将SR中记录由小到大地并入TR
        if (SR[i].key<=SR[j].key)  
            TR[k] = SR[i++];
        else 
            TR[k] = SR[j++];
    }
    if (i<=m) // 将剩余的 SR[i..m] 复制到 TR
        TR[k..n] = SR[i..m];
    if (j<=n) // 将剩余的 SR[j..n] 复制到 TR 
        TR[k..n] = SR[j..n];
} // Merge

void Msort ( RcdType SR[],  
            RcdType &TR1[], int s, int t ) {
    // 将SR[s..t] 归并排序为 TR1[s..t]
    if (s==t) 
        TR1[s]=SR[s];
    else {
        m = (s+t)/2;
        // 将SR[s..t]平分为SR[s..m]和SR[m+1..t]
        Msort (SR, TR2, s, m);
        // 递归地将SR[s..m]归并为有序的TR2[s..m]
        Msort (SR, TR2, m+1, t);
        //递归地SR[m+1..t]归并为有序的TR2[m+1..t]
        Merge (TR2, TR1, s, m, t);
        // 将TR2[s..m]和TR2[m+1..t]归并到TR1[s..t]
    }
} // Msort

void MergeSort (SqList &L) {
   // 对顺序表 L 作2-路归并排序
   MSort(L.r, L.r, 1, L.length);
} // MergeSort
```

容易看出，对 n 个记录进行归并排序的时间复杂度为Ο(nlogn)。

即：每一趟归并的时间复杂度为 O(n)，总共需进行 [log~2~n] 趟。

### 基数排序

基数排序是一种借助“多关键字排序”的思想来实现“单关键字排序”的内部排序算法。

#### 多关键字的排序

n 个记录的序列  { R1, R2, …，Rn}对关键字 (Ki0, Ki1,…,Kid-1) 有序是指：

对于序列中任意两个记录 Ri 和 Rj (1≤i<j≤n) 都满足下列(词典)有序关系：(Ki0, Ki1, …,Kid-1) <  (Kj0, Kj1, …,Kjd-1)

其中: K0被称为 “最主”位关键字，Kd-1 被称为 “最次”位关键字。

##### 最高位优先MSD法

先对K0进行排序，并按 K0 的不同值将记录序列分成若干子序列之后，分别对 K1 进行排序，...…，依次类推，直至最后对最次位关键字排序完成为止。

##### 最低位优先LSD法

先对 Kd-1 进行排序，然后对 Kd-2   进行排序，依次类推，直至对最主位关键字 K0排序完成为止。

#### 链式基数排序

假如多关键字的记录序列中，每个关键字的取值范围相同，则按LSD法进行排序时，可以采用“分配-收集”的方法，其好处是不需要进行关键字间的比较。

对于数字型或字符型的单关键字，可以看成是由多个数位或多个字符构成的多关键字，此时可以采用这种“分配-收集”的办法进行排序，称作基数排序法。

例如对数字按个位数、十位数、百位数依次分配-收集。

在计算机上实现基数排序时，为减少所需辅助存储空间，应采用链表作存储结构，即链式基数排序，具体作法为：

1. 待排序记录以指针相链，构成一个链表
2. “分配” 时，按当前“关键字位”所取值，将记录分配到不同的 “链队列” 中，每个队列中记录的 “关键字位” 相同；
3. “收集”时，按当前关键字位取值从小到大将各队列首尾相链成一个链表;
4. 对每个关键字位均重复2) 和3) 两步。

注意：

1. “分配”和“收集”的实际操作仅为修改静态链表中的指针和设置队列的头、尾指针；
2. 为查找使用，该静态链表尚需应用算法Arrange 将它调整为有序表。

基数排序的时间复杂度为O(d(n+rd))。

其中：分配为O(n)

​      　  收集为O(rd)(rd为“基”)

​      　  d为“分配-收集”的趟数

### 各种排序方法的综合比较

#### 时间性能

1. 平均的时间性能

   | 时间复杂度 | 排序方法                             |
   | ---------- | ------------------------------------ |
   | O(nlogn)   | 快速排序、堆排序和归并排序           |
   | O(n^2^)    | 直接插入排序、起泡排序和简单选择排序 |
   | O(n)       | 基数排序                             |

2. 当待排记录序列按关键字顺序有序时

   直接插入排序和起泡排序能达到O(n)的时间复杂度

   快速排序的时间性能蜕化为O(n^2^) 。

3. 简单选择排序、堆排序和归并排序的时间性能不随记录序列中关键字的分布而改变。

#### 空间性能

指的是排序过程中所需的辅助空间大小

1. 所有的简单排序方法(包括：直接插入、起泡和简单选择) 和堆排序的空间复杂度为O(1)；
2. 快速排序为O(logn)，为递归程序执行过程中，栈所需的辅助空间；
3. 归并排序所需辅助空间最多，其空间复杂度为 O(n);
4. 链式基数排序需附设队列首尾指针，则空间复杂度为 O(rd)。

#### 稳定性能

稳定的排序方法指的是，对于两个关键字相等的记录，它们在序列中的相对位置，在排序之前和经过排序之后，没有改变。

当对多关键字的记录序列进行LSD方法排序时，必须采用稳定的排序方法。

对于不稳定的排序方法，只要能举出一个实例说明即可。

快速排序、堆排序和希尔排序是不稳定的排序方法。

#### 时间复杂度的下限

本章讨论的各种排序方法，除基数排序外，其它方法都是基于“比较关键字”进行排序的排序方法。可以证明，  这类排序法可能达到的最快的时间复杂度为O(nlogn)。  (基数排序不是基于“比较关键字”的排序方法，所以它不受这个限制。)

### 外部排序

1. 待排序的记录数量很大，不能一次装入内存，则无法利用前几节讨论的排序方法 (否则将引起频繁访问外存)；
2. 对外存中数据的读/写是以“数据块”为单位进行的；读/写外存中一个“数据块”的数据所需要的时间为：

$$
T_{I/O} = t_{seek} + t_{la} + n\times t_{wm}
$$

​       其中

​       t~seek~  为寻查时间(查找该数据块所在磁道)

​       t~la~  为等待(延迟)时间

​       n $\times$ t~wm~ 为传输数据块中n个记录的时间。

**基本过程**

1. 按可用内存大小，利用内部排序方法，构造若干( 记录的) 有序子序列，通常称外存中这些记录有序子序列为“归并段”；
2. 通过“归并”，逐步扩大 (记录的)有序子序列的长度，直至外存中整个记录序列按关键字有序为止。

一般情况下，假设待排记录序列含 m 个初始归并段，外排时采用 k-路归并，则归并趟数为 [log~k~m]，显然，随着k的增大归并的趟数将减少，因此对外排而言，通常采用多路归并。





<div style="height:300px"></div>