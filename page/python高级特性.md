###### [返回主页](../README.md)

------



# python高级

## 1 迭代器

- 迭代是Python最强大的功能之一，是访问集合元素的一种方式。
- 迭代器是一个可以记住遍历的位置的对象。
- 迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。

### 1.1 迭代器基本方法

#### iter()

```python
list=[1,2,3,4]
# 创建迭代器对象
it = iter(list)
```

#### next()

```python
# 输出迭代器的下一个元素
print (next(it)) # 1
print (next(it)) # 2

for x in it:
    print (x, end=" ")
# 1 2 3 4
```

### 1.2 为自定义类创建迭代器

- 把一个类作为一个迭代器使用需要在类中实现两个方法`__iter__()`与`__next__()`
- `__iter__()`方法返回一个特殊的迭代器对象， 这个迭代器对象实现了`__next__()`方法并通过 StopIteration 异常标识迭代的完成。

```python
class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    if self.a <= 20:
      x = self.a
      self.a += 1
      return x
    else:
      raise StopIteration

myclass = MyNumbers()
myiter = iter(myclass)

for x in myiter:
  print(x) # 1-20
```

## 2 生成器

### 2.1 生成器的介绍

根据程序员制定的规则循环生成数据，当条件不成立时则生成数据结束。数据不是一次性全部生成处理，而是使用一个，再生成一个，可以节约大量的内存。

### 2.2 创建生成器

#### 生成器推导式

- 与列表推导式类似，只不过生成器推导式使用**小括号**
- next 函数获取生成器中的下一个值
- for 循环遍历生成器中的每一个值

```python
# 创建生成器
my_generator = (i * 2 for i in range(5))
print(my_generator) # <generator object <genexpr> at 0x101367048>

# next获取生成器下一个值
value = next(my_generator)
print(value) # 0

# 遍历生成器

for value in my_generator:
    print(value) # 2 4 6 8

```

#### yield 关键字

- 只要在def函数里面看到有 yield 关键字那么就是生成器
- 代码执行到 yield 会暂停，然后把结果返回出去，下次启动生成器会在暂停的位置继续往下执行
- 生成器如果把数据生成完成，再次获取生成器中的下一个数据会抛出一个StopIteration 异常，表示停止迭代异常
- while 循环内部没有处理异常操作，需要手动添加处理异常操作
- for 循环内部自动处理了停止迭代异常，使用起来更加方便

```python
def mygenerater(n):
    for i in range(n):
        print('开始生成...')
        yield i
        print('完成一次...')

if __name__ == '__main__':

    g = mygenerater(2)

    # 获取生成器中下一个值
    result = next(g)
    print(result)

    while True:
        try:
            result = next(g)
            print(result)
        except StopIteration as e:
            break

    # for遍历生成器, for 循环内部自动处理了停止迭代异常，使用起来更加方便
    for i in g:
        print(i)
```

## 3 闭包

### 3.1 闭包简介

我们前面已经学过了函数，我们知道当函数调用完，函数内定义的变量都销毁了，但是我们有时候需要保存函数内的这个变量，每次在这个变量的基础上完成一些列的操作，比如: 每次在这个变量的基础上和其它数字进行求和计算，那怎么办呢?

我们就可以通过咱们今天学习的闭包来解决这个需求。

#### 闭包定义

在函数嵌套的前提下，内部函数使用了外部函数的变量，并且外部函数返回了内部函数，我们把这个使用外部函数变量的内部函数称为闭包。

### 3.2 闭包的构成条件

- 在函数嵌套(函数里面再定义函数)的前提下
- 内部函数使用了外部函数的变量(还包括外部函数的参数)
- 外部函数返回了内部函数

```python
# 定义一个外部函数
def func_out(num1):
    # 定义一个内部函数
    def func_inner(num2):
        # 内部函数使用了外部函数的变量(num1)
        result = num1 + num2
        print("结果是:", result)
    # 外部函数返回了内部函数，这里返回的内部函数就是闭包
    return func_inner

# 创建闭包实例
f = func_out(1)
# 执行闭包
f(2)  # 结果是: 3
f(3)  # 结果是: 4
```

**通过上面的输出结果可以看出闭包保存了外部函数内的变量num1，每次执行闭包都是在num1 = 1 基础上进行计算。**

### 3.3 闭包的作用

- 闭包可以保存外部函数内的变量，不会随着外部函数调用完而销毁。
- 由于闭包引用了外部函数的变量，则外部函数的变量没有及时释放，消耗内存。

### 3.4 修改闭包内使用的外部变量

- 如果直接给num1赋新值，则会在内部函数中建立一个新的局部变量
- 可使用 `nonlocal num1`告诉解释器，此处使用的是外部变量，再给num1赋新值

## 4 装饰器

### 4.1 装饰器的定义

- 给已有函数增加额外功能的函数，它本质上就是一个闭包函数
- 装饰器的功能特点
  - 不修改已有函数的源代码
  - 不修改已有函数的调用方式
  - 给已有函数增加额外的功能

### 4.2 装饰器的实现

#### 装饰器的闭包函数写法

```python
# 装饰器的基本雏形
def decorator(fn): # fn:目标函数.
    def inner():
        '''执行函数之前'''
        fn() # 执行被装饰的函数
        '''执行函数之后'''
    return inner
```

- 闭包函数有且只有一个参数，必须是函数类型，这样定义的函数才是装饰器。
- 写代码要遵循开放封闭原则，它规定已经实现的功能代码不允许被修改，但可以被扩展。

#### 装饰器的语法糖写法

Python给提供了一个装饰函数更加简单的写法，那就是语法糖，语法糖的书写格式是: @装饰器名字，通过语法糖的方式也可以完成对已有函数的装饰

```python
# 添加一个登录验证的功能
def check(fn):
    print("装饰器函数执行了")
    def inner():
        print("请先登录....")
        fn()
    return inner

# 使用语法糖方式来装饰函数
@check
def comment():
    print("发表评论")


comment()
# 请先登录....
# 发表评论
```

- @check 等价于 comment = check(comment)
- 装饰器的执行时间是加载模块时立即执行。

### 4.3 装饰器举例

#### 装饰带有参数的函数

```python
# 添加输出日志的功能
def logging(fn):
    def inner(num1, num2):
        print("--正在努力计算--")
        fn(num1, num2)

    return inner


# 使用装饰器装饰函数
@logging
def sum_num(a, b):
    result = a + b
    print(result)


sum_num(1, 2)
# --正在努力计算--
# 3
```

#### 装饰带有返回值的函数

```python
# 添加输出日志的功能
def logging(fn):
    def inner(num1, num2):
        print("--正在努力计算--")
        result = fn(num1, num2)
        return result
    return inner


# 使用装饰器装饰函数
@logging
def sum_num(a, b):
    result = a + b
    return result


result = sum_num(1, 2)
print(result)
# --正在努力计算--
# 3
```

#### 装饰带有不定长参数的函数

```python
# 添加输出日志的功能
def logging(fn):
    def inner(*args, **kwargs):
        print("--正在努力计算--")
        fn(*args, **kwargs)

    return inner


# 使用语法糖装饰函数
@logging
def sum_num(*args, **kwargs):
    result = 0
    for value in args:
        result += value

    for value in kwargs.values():
        result += value

    print(result)

sum_num(1, 2, a=10)
# --正在努力计算--
# 13
```

#### 装饰器通用语法格式

```python
# 通用装饰器
def logging(fn):
  def inner(*args, **kwargs):
      print("--正在努力计算--")
      result = fn(*args, **kwargs)
      return result

  return inner
```

## 5 property属性

### 5.1 property属性的介绍

- property属性就是负责把一个方法当做属性进行使用，这样做可以简化代码使用。
- 定义property属性有两种方式
  - 装饰器方式
  - 类属性方式

#### 装饰器方式

- `@property`表示把方法当做属性使用, 表示当获取属性时会执行下面修饰的方法
- `@方法名.setter`表示把方法当做属性使用,表示当设置属性时会执行下面修饰的方法
- 装饰器方式的property属性修饰的方法名一定要一样。

```python
class Person(object):

    def __init__(self):
        self.__age = 0

    # 装饰器方式的property, 把age方法当做属性使用, 表示当获取属性时会执行下面修饰的方法
    @property
    def age(self):
        return self.__age

    # 把age方法当做属性使用, 表示当设置属性时会执行下面修饰的方法
    @age.setter
    def age(self, new_age):
        if new_age >= 150:
            print("成精了")
        else:
            self.__age = new_age

# 创建person
p = Person()
print(p.age)
p.age = 100
print(p.age)
p.age = 1000
# 0
# 100
# 成精了
```

#### 类属性方式

- property的参数说明:
  - 第一个参数是获取属性时要执行的方法
  - 第二个参数是设置属性时要执行的方法

```python
class Person(object):

    def __init__(self):
        self.__age = 0

    def get_age(self):
        """当获取age属性的时候会执行该方法"""
        return self.__age

    def set_age(self, new_age):
        """当设置age属性的时候会执行该方法"""
        if new_age >= 150:
            print("成精了")
        else:
            self.__age = new_age

    # 类属性方式的property属性
    age = property(get_age, set_age)

# 创建person
p = Person()
print(p.age)
p.age = 100
print(p.age)
p.age = 1000
# 0
# 100
# 成精了
```

## 6 with语句

### 6.1 with语句的使用

Python提供了 with 语句，既简单又安全，并且 with 语句执行完成以后自动调用关闭文件操作，即使出现异常也会自动调用关闭文件操作

```python
# 1、以写的方式打开文件
with open("1.txt", "w") as f:
    # 2、读取文件内容
    f.write("hello world")
```

### 6.2 上下文管理器

- 一个类只要实现了__enter__()和__exit__()这个两个方法，通过该类创建的对象我们就称之为上下文管理器。
- 上下文管理器可以使用 with 语句，with语句之所以这么强大，背后是由上下文管理器做支撑的，也就是说刚才使用 open 函数创建的文件对象就是就是一个上下文管理器对象。

#### 自定义上下文管理器类,模拟文件操作

```python
class File(object):

    # 初始化方法
    def __init__(self, file_name, file_model):
        # 定义变量保存文件名和打开模式
        self.file_name = file_name
        self.file_model = file_model

    # 上文方法
    def __enter__(self):
        print("进入上文方法")
        # 返回文件资源
        self.file = open(self.file_name,self.file_model)
        return self.file

    # 下文方法
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("进入下文方法")
        self.file.close()


if __name__ == '__main__':

    # 使用with管理文件
    with File("1.txt", "r") as file:
        file_data = file.read()
        print(file_data)

# 进入上文方法
# hello world
# 进入下文方法

```

- `__enter__`表示上文方法，需要返回一个操作文件对象
- `__exit__`表示下文方法，with语句执行完成会自动执行，即使出现异常也会执行该方法。

#### 上下文管理器的装饰器实现方式

假如想要让一个函数成为上下文管理器，Python 还提供了一个 `@contextmanager`的装饰器，更进一步简化了上下文管理器的实现方式。通过 yield 将函数分割成两部分，yield 上面的语句在 `__enter__` 方法中执行，yield 下面的语句在 `__exit__` 方法中执行，紧跟在 yield 后面的参数是函数的返回值。

```python
# 导入装饰器
from contextlib import contextmanager


# 装饰器装饰函数，让其称为一个上下文管理器对象
@contextmanager
def my_open(path, mode):
    try:
        # 打开文件
        file = open(file_name, file_mode)
        # yield之前的代码好比是上文方法
        yield file
    except Exception as e:
        print(e)
    finally:
        print("over")
        # yield下面的代码好比是下文方法
        file.close()

# 使用with语句
with my_open('out.txt', 'w') as f:
    f.write("hello , the simplest context manager")
```

## 7 拷贝

### 7.1 浅拷贝

copy函数是浅拷贝，只对可变类型的第一层对象进行拷贝，对拷贝的对象开辟新的内存空间进行存储，不会拷贝对象内部的子对象。

#### 不可变类型的浅拷贝

不可变类型进行浅拷贝:**不会给拷贝的对象开辟新的内存空间，而只是拷贝了这个对象的引用**

```python
import copy  # 使用浅拷贝需要导入copy模块

# 不可变类型有: 数字、字符串、元组

a1 = 123123
b1 = copy.copy(a1)  # 使用copy模块里的copy()函数就是浅拷贝了
# 查看内存地址
print(id(a1))
print(id(b1))

print("-" * 10)
a2 = "abc"
b2 = copy.copy(a2)
# 查看内存地址
print(id(a2))
print(id(b2))

print("-" * 10)
a3 = (1, 2, ["hello", "world"])
b3 = copy.copy(a3)
# 查看内存地址
print(id(a3))
print(id(b3))

# 运行结果:
# 140459558944048
# 140459558944048
# ----------
# 140459558648776
# 140459558648776
# ----------
# 140459558073328
# 140459558073328
```

#### 可变类型的浅拷贝

可变类型进行浅拷贝:**只对可变类型的第一层对象进行拷贝，对拷贝的对象会开辟新的内存空间进行存储，子对象不进行拷贝。**

```python
import copy # 使用浅拷贝需要导入copy模块

# 可变类型有: 列表、字典、集合

a1 = [1, 2]
b1 = copy.copy(a1) # 使用copy模块里的copy()函数就是浅拷贝了
# 查看内存地址
print(id(a1))
print(id(b1))
print("-" * 10)
a2 = {"name": "张三", "age": 20}
b2 = copy.copy(a2)
# 查看内存地址
print(id(a2))
print(id(b2))
print("-" * 10)
a3 = {1, 2, "王五"}
b3 = copy.copy(a3)
# 查看内存地址
print(id(a3))
print(id(b3))

print("-" * 10)
a4 = [1, 2, [4, 5]]
# 注意：浅拷贝只会拷贝父对象，不会对子对象进行拷贝
b4 = copy.copy(a4) # 使用copy模块里的copy()函数就是浅拷贝了
# 查看内存地址
print(id(a4))
print(id(b4))
print("-" * 10)
# 查看内存地址
print(id(a4[2]))
print(id(b4[2]))

# 修改数据
a4[2][0] = 6

# 子对象的数据会受影响
print(a4)
print(b4)


# 运行结果:
# 139882899585608
# 139882899585800
# ----------
# 139882919626432
# 139882919626504
# ----------
# 139882919321672
# 139882899616264
# ----------
# 139882899587016
# 139882899586952
# ----------
# 139882899693640
# 139882899693640
# [1, 2, [6, 5]]
# [1, 2, [6, 5]]
```

### 7.2 深拷贝

deepcopy函数是深拷贝, 只要发现对象有可变类型就会对该对象到最后一个可变类型的每一层对象就行拷贝, 对每一层拷贝的对象都会开辟新的内存空间进行存储。

#### 不可变类型的深拷贝

不可变类型进行深拷贝:**如果子对象没有可变类型则不会进行拷贝，而只是拷贝了这个对象的引用，否则会对该对象到最后一个可变类型的每一层对象就行拷贝, 对每一层拷贝的对象都会开辟新的内存空间进行存储**

```python
import copy  # 使用深拷贝需要导入copy模块

# 可变类型有: 列表、字典、集合

a1 = [1, 2]
b1 = copy.deepcopy(a1)  # 使用copy模块里的deepcopy()函数就是深拷贝了
# 查看内存地址
print(id(a1))
print(id(b1))
print("-" * 10)
a2 = {"name": "张三"}
b2 = copy.deepcopy(a2)
# 查看内存地址
print(id(a2))
print(id(b2))
print("-" * 10)
a3 = {1, 2}
b3 = copy.deepcopy(a3)
# 查看内存地址
print(id(a3))
print(id(b3))
print("-" * 10)

a4 = [1, 2, ["李四", "王五"]]
b4 = copy.deepcopy(a4)  # 使用copy模块里的deepcopy()函数就是深拷贝了
# 查看内存地址
print(id(a4))
print(id(b4))

# 查看内存地址
print(id(a4[2]))
print(id(b4[2]))
a4[2][0] = "王五"
# 因为列表的内存地址不同，所以数据不会收到影响
print(a4)
print(b4)

# 运行结果:
# 140348291721736
# 140348291721928
# ----------
# 140348311762624
# 140348311221592
# ----------
# 140348311457864
# 140348291752456
# ----------
# 140348291723080
# 140348291723144
# 140348291723208
# 140348291723016
# [1, 2, ['王五', '王五']]
# [1, 2, ['李四', '王五']]
```

#### 可变类型的深拷贝

可变类型进行深拷贝:**对该对象到最后一个可变类型的每一层对象就行拷贝, 对每一层拷贝的对象都会开辟新的内存空间进行存储。**

### 7.3 拷贝总结

- 浅拷贝使用`copy.copy`函数
- 深拷贝使用`copy.deepcopy`函数
- 不管是给对象进行深拷贝还是浅拷贝，只要拷贝成功就会开辟新的内存空间存储拷贝的对象。
- 浅拷贝和深拷贝的区别是:
  - 浅拷贝最多拷贝对象的一层
  - 深拷贝可能拷贝对象的多层。
