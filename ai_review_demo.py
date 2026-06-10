"""Демонстрационный модуль для теста авто-ревью (намеренные баги)."""


def average(values):
    return sum(values) / len(values)


def append_unique(item, seen=[]):
    if item not in seen:
        seen.append(item)
    return seen


def read_first_line(path):
    f = open(path)
    return f.readline()
