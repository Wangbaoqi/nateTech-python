
# 1.成员运算
# x in s
# x not in s

name = "Alice"

str = "Hello, Alice!"

print(name in str)  # True
print(name not in str)  # False

# 2.连接运算
# x + x
# x * n

print(name + str)
print(name * 3)

# 3.切片操作

string = "Hello, World!"

string[0]  # 'H'
string[0:5]  # 'Hello'
string[7:12]  # 'World'

# 4. 字符串方法
# 4.1 字符串中子串出现的次数
string.count("o")  # 2

# 4.2 字符串中所有的字符都是字母或数字
'123'.isalnum()  # True
'123a'.isalnum()  # True
'123!'.isalnum()  # False

print('123'.isalnum())  # True
print('123a'.isalnum())  # True
print('123!'.isalnum())  # False

# 4.3 字符串中所有的字符都是字母
'abc'.isalpha()  # True
'abc123'.isalpha()  # False

# 4.4 字符串中所有的字符都是数字
'123'.isdigit()  # True
'123a'.isdigit()  # False

# 4.5 字符串中所有的字符都是小写字母
'abc'.islower()  # True
'abc123'.islower()  # True
'ABC'.islower()  # False

# 4.6 字符串中所有的字符都是大写字母
'ABC'.isupper()  # True
'ABC123'.isupper()  # True

# 4.7 字符串中所有的字符都是空白字符
'   '.isspace()  # True

# 4.8 字符串中所有的字符都是标题化的
'Hello World'.istitle()  # True
'Hello world'.istitle()  # False

# 4.9 join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串
print(' '.join(string))  # 'H e l l o ,   W o r l d !'
print('-'.join(string))  # 'H-e-l-l-o-,---W-o-r-l-d-!'