import os

def describe_col_uni_nuni(df, directory=None):
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 경로 지정
    file_path = os.path.join(directory, 'df_summary.txt')

    # 파일 쓰기
    with open(file_path, 'w') as f:
        for i in range(len(df.columns)):
            f.write(f'{i + 1}: {df.columns[i]}:\nnunique: ' +
                    str(df[df.columns[i]].nunique()) + "\n" +
                    str(df[df.columns[i]].unique()) + "\n\n")


def describe_col(df, directory=None):
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 경로 지정
    file_path = os.path.join(directory, 'df_col_summary.txt')

    # 파일 쓰기
    with open(file_path, 'w') as f:
        for i in range(len(df.columns)):
            f.write(f'{i + 1}: {df.columns[i]}:\n\n')
