binary_classification_sys_prompt="""You are a discerning rationalist."""
# binary_classification_sys_prompt="""You are a helpful assistant."""

binary_classification_prompt="""You are given a first-hand account of information from a meme. You are given different pieces of information to figure out whether the meme is hateful (H) or not (N) after reasoning why you chose what you did.

An example output would be like the following. Following this format is VERY important:

```
These are my reasons why the meme I'm given isn't hateful.
###
N
```

A meme is hateful if it is a  direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. We define attack as violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hate speech.

This is background information that might be relevant for understanding the meme:
{background}


This is the information you are given about the meme:


```
{meme_info}
```

Reason and do critical thinking to verify if it is hateful or not and be careful to see if it fits the definition of hateful. After your reasoning, finish by writing three hashtags (i.e. ###) to separate the reasoning and the answer and ONLY outputting 'H' if the meme is hateful or 'N' if it is not hateful. Following this format is VERY important.
"""



binary_classification_prompt1="""You are given a first-hand account of information from a meme. You are given different pieces of information to figure out whether the meme is hateful (H) or not (N) after reasoning why you chose what you did.

An example output would be like the following. Following this format is VERY important:

```
These are my reasons why the meme I'm given isn't hateful.
###
N
```

A meme is hateful if it is a  direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. We define attack as violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hate speech.
This is the information you are given about the meme:

```
{meme_info}
```

Reason and do critical thinking to verify if it is hateful or not and be careful to see if it fits the definition of hateful. After your reasoning, finish by writing three hashtags (i.e. ###) to separate the reasoning and the answer and ONLY outputting 'H' if the meme is hateful or 'N' if it is not hateful. Following this format is VERY important.
"""

binary_classification_prompt2="""You are given a second-hand account of information from a meme but not the meme itself. You are given different pieces of information from an AI that has no conception if a meme is hateful or not. You need to figure out whether the meme is hateful (H) or not (N) after reasoning and thinking out loud about the information you were given.

Remember a meme is hateful if it is a direct or indirect negative comment on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. We define hateful as dehumanizing (ex. comparing people to non-human things, e.g. animals) speech, statements of inferiority (ex. negative generalization),  perpetuating harmful biases and stereotypes,  and calls for exclusion or segregation. Mocking hate crime is also considered hate speech (ex. trivializing sensitive topics with jokes is a common form of this).

It is VERY important to follow the EXACT format for putting three dashes (---) and then giving the end result of ONLY 'N' or 'H'. I'll be using python to split the text in this format. This is an example of the analysis you will do:

### INPUT
```
{{'Text': 'this is what ignorant people think when they see a muslim.', 'Image Caption': 'It two images side by side of a muslim woman wearing a hijab and another of a muslim woman in army gear with a gun.', 'Surface Message': 'It two images side by side of a muslim woman wearing a hijab and another of a muslim woman in army gear with a gun. The author describes the image as this is what ignorant people think when they see a muslim.'}}
```

### BACKGROUND
````
'Familiarity with the religion of Islam and its followers, Muslims and discriminatory stereotypes against them.'
```

### ANSWER
```
Based on the information presented, the meme juxtaposes two distinct images of a Muslim woman: one in a Hijab and the other in military gear holding a gun. The text overlays stating 'this is what ignorant people think when they see a Muslim' suggest that it's criticizing stereotypes and misconceptions around Muslim women, both of which can be considered harmful stereotypes. 

It is vital to note the framing of these stereotypes as 'what ignorant people think,' indicating an intention not to endorse these stereotypes but to question and criticize them. It is thus an attempt to raise awareness of the issue of stereotyping. 

In the end, it's critical to see that this meme isn't direct hate speech towards a particular group or individual (based on race, religion, gender, etc.). 

Understanding the subtleties involved in this image requires background knowledge about stereotypes concerning Muslim women; therefore, it is relevant to consider this information in the analysis.

---
N
```

This is the input information you are given about the meme from the objective AI:
### INPUT
```
{meme_info}
```

### BACKGROUND
```
{background}
```

### OUTPUT
"""