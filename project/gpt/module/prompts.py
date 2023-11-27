binary_classification_sys_prompt="""You are a helpful assistant."""

binary_classification_prompt="""You are given a first-hand account of information from a meme. You are given different pieces of information to figure out whether the meme is hateful (H) or not (N). A meme is hateful if it is a  direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. We define attack as violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hate speech.

This is the information you are given about the meme:

```
{meme_info}
```

Reason and do critical thinking to verify if it is hateful or not and be careful to see if it fits the definition of hateful. After your reasoning, finish by writing three hashtags (i.e. ###) to separate the reasoning and the answer and ONLY outputting 'H' if the meme is hateful or 'N' if it is not hateful.
"""
