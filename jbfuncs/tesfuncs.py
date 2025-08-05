from jobflow import job

@job
def count_str(input_str: str):
    return len(input_str)

@job
def sum_numbers(numbers):
    print(sum(numbers))
    return sum(numbers)
