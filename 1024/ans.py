def split_command(command):
    command = command.strip('\n')
    flag = 1
    if command[0] == 'u':
        n = command[3:-1]
        if n == '':
            n = 1
        elif n[0] == '-':
            flag = -1
            n = n[1:]
    elif command[0] == 'd' or command[0] == 'l':
        n = command[5:-1]
        if n == '':
            n = 1
        elif n[0] == '-':
            flag = -1
            n = n[1:]
    elif command[0] == 'r':
        n = command[6:-1]
        if n == '':
            n = 1
        elif n[0] == '-':
            flag = -1
            n = n[1:]
    return command[0], flag * int(n)


def main():
    with open('./command.txt', 'r') as f:
        command = f.readlines()
    command = [line.split(';') for line in command]

    with open('./matrix.txt', 'r') as f:
        mat = f.readlines()
    mat = [list(line.strip('\n')) for line in mat]

    cur_row = 0
    cur_col = 0
    rows = len(mat) # 256
    cols = len(mat[0]) # 256
    for row in range(len(command)):
        for col in range(len(command[row])):
            com, n = split_command(command[row][col])
            if com == 'u':
                cur_row -= n
            elif com == 'd':
                cur_row += n
            elif com == 'l':
                cur_col -= n
            elif com == 'r':
                cur_col += n
            cur_row %= rows
            cur_col %= cols

    return cur_row+1, cur_col+1, mat[cur_row][cur_col]

if __name__ == '__main__':
    print(main())



