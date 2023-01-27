for i in range(1, 101):
    if i % 5 == 0 and i % 3 == 0: print(i, 'FizzBuzz')
    elif i % 5 == 0: print(i, 'buzz')
    elif i % 3 == 0: print(i, 'fizz')