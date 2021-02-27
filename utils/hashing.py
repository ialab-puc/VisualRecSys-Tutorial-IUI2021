"""Utilities to hash elements.

This module contains a class (HashesContainer) that calculate hashes
and notify collisions.
"""


def pre_hash(triple, contains_iter=True):
    """Prepare tuple to be hashed.

    This means that each element of the tuple will be converted to
    string. The first item (profile) should be iterable, whilst only
    the second and third items (positive and negative) will be
    considered as well, leaving the rest out of the tuple.

    Args:
        triple: Tuple with the profile items (iterable), positive
            item and its negative counterpart.
        contains_iter: Optional. If its true, sorts and transform
            each element of the first element in a truple with str.

    Returns:
        Same tuple but converted to string. Example:

            ([1, 2], 3, 4, 5)

        Becomes:

            (['1', '2'], '3', '4')

        If contains_iter is False:

            (1, 2, 3)

        Becomes:

            ('1', '2', '3')
    """
    if not contains_iter:
        return (str(triple[0]), str(triple[1]), str(triple[2]))
    _sorted_t0 = tuple(sorted([str(_id) for _id in triple[0]]))
    return (_sorted_t0, str(triple[1]), str(triple[2]))


class HashesContainer:
    """Manages hashes of elements to detect duplicates.

    A custom hashing function is used to hash an arbitrary number of
    elements. Also, stores used hashes to detect collisions and 
    count them.

    Attributes:
        collisions: Current count of hash collisions detected.
        hashes: Set of used hashes.
    """

    _MOD = 402653189
    _BASE = 92821

    def __init__(self):
        """Inits an empty HashesContainer"""
        self.collisions = 0
        self.hashes = set()

    def enroll(self, *content):
        """Tries to register a new hash and reports collision.

        Hahes new content and returns True if was added successfully
        (no collision). 

        Args:
            *content: Information to be hashed (must contain/be
                iterables and/or str)

        Returns:
            True if no hash collision was detected and False
            otherwise.
        """
        h = self.hash(*content)
        if h in self.hashes:
            self.collisions += 1
            return False
        self.hashes.add(h)
        return True

    def hash(self, *args, h=0):
        """Calculates hash of given elements.

        Uses a custom hash function to calculate hashes recursively.

        Args:
            *args: Information to be hashed (must contain/be
                iterables and/or str).
            h: Optional. Current hash value. Defaults to 0.

        Returns:
            Hash value as an integer.
        """
        for arg in args:
            if isinstance(arg, str):
                h = ((h * self._BASE) % self._MOD + int(arg, 32)) % self._MOD
            else:
                h = self.hash(*arg, h=h)
        return h
