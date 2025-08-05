from jobflow import job

@job
def time_website(website: str):
    import urllib.request
    from time import perf_counter

    with urllib.request.urlopen(website) as f:
        start_time = perf_counter()
        f.read()
        end_time = perf_counter()

    return end_time - start_time

@job
def sum_numbers(numbers):
    return sum(numbers)
