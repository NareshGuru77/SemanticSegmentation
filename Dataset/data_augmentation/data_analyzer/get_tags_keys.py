import collections


_PERCENTAGE_KEY = 'Percentage of pixels'
_COUNT_KEY = 'Class count'
_WEIGHT_KEY = 'Class weight'

_TRAINING_TAG = 'TrainingSet'
_VALIDATION_TAG = 'ValidationSet'
_TEST_TAG = 'TestSet'

_TAG_SEPARATOR = '/'

_DATA_KEY = 'data'
_INFO_KEY = 'info'


class TagsKeys(
    collections.namedtuple('tags', [
        'percentage_key',
        'count_key',
        'weight_key',
        'training_tag',
        'validation_tag',
        'test_tag',
        'tag_separator',
        'data_key',
        'info_key',
        ])):

    __slots__ = ()

    def __new__(cls):

        return super(TagsKeys, cls).__new__(
            cls, _PERCENTAGE_KEY, _COUNT_KEY, _WEIGHT_KEY,
            _TRAINING_TAG, _VALIDATION_TAG, _TEST_TAG,
            _TAG_SEPARATOR, _DATA_KEY, _INFO_KEY)


tags_keys = TagsKeys()