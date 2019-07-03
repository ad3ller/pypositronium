""" misc. tools
"""


def where(function, iterable):
    """ Elements where function mapped to iterable evaluates as True.

    args:
        function :: function
        iterable :: list

    return:
        elements :: generator
    """
    return (x for x in iterable if function(x))


def argwhere(function, iterable):
    """ Indexes where function mapped to iterable evaluates as True.

    args:
        function :: function
        iterable :: list

    return:
        indexes :: generator
    """
    return (i for i, x in enumerate(iterable) if function(x))
