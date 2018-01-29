---
layout: post
title: 笔记
date: 2018-01-29
---
1.
AWT无法实现跨平台统一的GUI显示，因此出现了Swing，Swing是AWT的增强版，但是并不能完全代替AWT，这两种组件有时需要同时出现在一个用户图形界面中
2.
通常优先级由高到低的顺序依次是：
- 增量和减量运算
- 算术运算
- 比较运算
- 逻辑运算
- 赋值运算
3.
子类继承父类后，当子类想要通过构造函数创建实例时，必须先实现父类的构造函数，如下

``` java
class Parent { // 父类
	Parent() {
		System.out.println("调用父类的parent()构造方法");
	}
}

class SubParent extends Parent { // 继承Parent类
	SubParent() {
		System.out.println("调用子类的SubParent()构造方法");
	}
}

public class Subroutine extends SubParent { // 继承SubParent类
	Subroutine() {
		System.out.println("调用子类的Subroutine()构造方法");
	}
	
	public static void main(String[] args) {
		Subroutine s = new Subroutine(); // 实例化子类对象
	}
}
```
结果如下图:
![](https://github.com/shencunzailaozhang/shencunzailaozhang.github.io/raw/master/assets/images/capture.PNG)
4.
java连接数据库时，首先需要加载数据库驱动，然后通过getConnection方法来连接数据库
