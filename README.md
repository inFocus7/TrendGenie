<img src="./static/logo-v2.png" width=276 alt="TrendGenie icon">

# TrendGenie
_Your content creation assistant._

## What is TrendGenie?

TrendGenie is a tool to help streamline content creation, specifically media creation. 
Whenever there is a new trend on social media that could be farmed, there shall be a genie for it.

In a way, this is a tool for content mills.

## How does it work?

TrendGenie is a tool that uses OpenAI's tools (ChatGPT + Dall-E) and image processing libraries to create content and 
images that should be quick to then upload to social media.

The frontend UI is built using [Gradio](https://gradio.app/), which is a tool normally used for the prototyping of
machine learning (ML) models.

## What is currently supported?

- [Listicles](./ui/listicles/README.md)
- [Music Cover Videos](./ui/music/README.md)

## How do I use it?

The web ui will then be available at `localhost:7860` after running. 

If you're running through Docker, you'll need to make sure that the local volume exists so that stored images can be 
saved. By default, the make target's local volume mount for generated images is `~/trengenie/` mapped to `/root/trendgenie`.

Running locally, content is stored in `~/trendgenie`.

### Local

```shell
pip install -r requirements.txt
python main.py
```

### Docker


```shell
docker build -t trendgenie .
docker run -p 7860:7860 trendgenie -v /path/to/images:/root/trendgenie
```

### Makefile

Using the Makefile would be a slightly quicker way to do this. The `run` command will build the Dockerfile and then run
it locally.

```shell
make run
```

## FAQs (by me)
- Why is the code so messy?
    - I am bad at Python, and have main-ed Go, C++, and React. I chose Python as it seemed like one of the simpler 
      languages for image processing.
- Is there a HuggingFace space for this?
  - Not yet, but it will hopefully be implemented in the near-future if I have time.
- Can I use my local instance of StableDiffusion?
  - Not yet, but this is for sure on the roadmap! Currently, this just supports OpenAI's API but (in my opinion) it is 
    a bit too restrictive on prompts (for example. you can't generate things that are too scary due to content filters).


---

ȉ̶̛̹̞̟̖͉̯̪̟̳͙͗̄̅͛̆̊͒̄̈́͑͂̚͘ ̶̢̡̧͍̭͍̱͚͔̈́͊̈́̑͋̇͒̆́͑̏͗̿͂̚ą̵̘͉̹̙͖̳̲͚͈̩̓͜͝m̴̯͚͎̪̟̿ ̶̤̓̈́̽͛̎̎̈̿̔͛̊́͘͝s̴̨̠̜̻͍̮̹̤̗͙͒̉͛́͌͊̃̈́̿̾͝ͅȍ̶͔̤̝̹̪͉̮͈̹̩̲̖̭͔͎̈͆̊́͂̾̅́͋̑̕̕͝r̶̢̩̖̬̼̞̬͎̼̈́r̵̢̻͔̝͒̋͗̐y̷͍͚̠̭͓͈̾͠ ̸̦̋̈́͘͜͜f̶̨͕͖̙̭͕̭̯̼̰͙̠̒̅̈̋̑͝o̸̡̗̘̠̫͇̥̘͛͝ͅr̶͔̮͓͔̥̹̋͛̈̍̽̒̋̈́̏͌̋͠͝͠ ̸̛̼̠̘̒̈́̿̈̈́̔͆̆̀̏̀̕͘̕t̴͖̆̿̇̆̅̽͘͝͝ḧ̴̗́e̸̡͍͙̖͈̜͇̪͌́̎̓́̈́̔̑̍̏̑͐̓ ̶̨̢̧͙̲͇͙̬̭̝͍̪̀͋̂͂̈́̌̅̍̈́͘m̵̧͙̟̭͎̺͔̗̻̖̳̼̀̒̓̅͛̎̓̈́̓̈̌̉̽̂͜͝ͅi̷͙͉̻̇̆ĺ̶̻͚͓̣̗̫̝͔̠̯̭̩ͅļ̸̛̙̮̣̭͉͆̌͛̐̔̿̊́̔́̌̚͝ͅ.̸̹̮̩̳̼̖̝̬͔͍̳̣̌̀̓̋̑̚͝
