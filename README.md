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

## Getting Started

The web ui will then be available at `localhost:7860` after running. 

If you're running through Docker, you'll need to make sure that the local volume exists so that stored images can be 
saved. By default, the make target's local volume mount for generated images is `~/trengenie/` mapped to `/root/trendgenie`.
Aside from allowing images to sync, the `~/trendgenie` directory is used to share/use fonts. Any font you want to use
should be placed in `~/trendgenie/fonts/`.

Running locally, content is stored in `~/trendgenie`.

### Local

```shell
pip install -r requirements.txt
python main.py
```

### Docker


```shell
docker build -t trendgenie .
docker run -p 7860:7860 trendgenie -v /path/to/trendgenie:/root/trendgenie
```

#### Volumes

Before starting the container, you'll need to make sure that the local volume exists so that stored images can be
mapped to the local filesystem.

### Makefile

Using the Makefile would be a slightly quicker way to do this. The `start` command will build the Dockerfile and then 
run it locally.

```shell
make start
```

#### Volumes

Before starting the container, you'll need to make sure that the local volume exists so that stored images can be
mapped to the local filesystem.

## FAQs
- Why is the code so messy?
    - I am bad at Python, and have main-ed Go, C++, and React. I chose Python as it seemed like one of the simpler 
      languages for image processing.
- Is there a HuggingFace space for this?
  - Not yet, but it will hopefully be implemented in the near-future if I have time.
- Can I use my local instance of StableDiffusion?
  - Not yet, but this is for sure on the roadmap! Currently, this just supports OpenAI's API but (in my opinion) it is 
    a bit too restrictive on prompts (for example. you can't generate things that are too scary due to content filters).
- Why don't I have any fonts?
  - If you ran this through Docker, you'll need to make sure that the local volume exists. In the local volume, you'll 
    need to create a `fonts` directory and place any fonts you want to use in there. The `fonts` directory is mapped to
    `/root/trendgenie/fonts` in the container where they are read.
  - I decided against shipping fonts with the container as it would make the container size much larger, 
    and I didn't want to have to deal with licensing issues.

---

ȉ̶̛̹̞̟̖͉̯̪̟̳͙͗̄̅͛̆̊͒̄̈́͑͂̚͘ ̶̢̡̧͍̭͍̱͚͔̈́͊̈́̑͋̇͒̆́͑̏͗̿͂̚ą̵̘͉̹̙͖̳̲͚͈̩̓͜͝m̴̯͚͎̪̟̿ ̶̤̓̈́̽͛̎̎̈̿̔͛̊́͘͝s̴̨̠̜̻͍̮̹̤̗͙͒̉͛́͌͊̃̈́̿̾͝ͅȍ̶͔̤̝̹̪͉̮͈̹̩̲̖̭͔͎̈͆̊́͂̾̅́͋̑̕̕͝r̶̢̩̖̬̼̞̬͎̼̈́r̵̢̻͔̝͒̋͗̐y̷͍͚̠̭͓͈̾͠ ̸̦̋̈́͘͜͜f̶̨͕͖̙̭͕̭̯̼̰͙̠̒̅̈̋̑͝o̸̡̗̘̠̫͇̥̘͛͝ͅr̶͔̮͓͔̥̹̋͛̈̍̽̒̋̈́̏͌̋͠͝͠ ̸̛̼̠̘̒̈́̿̈̈́̔͆̆̀̏̀̕͘̕t̴͖̆̿̇̆̅̽͘͝͝ḧ̴̗́e̸̡͍͙̖͈̜͇̪͌́̎̓́̈́̔̑̍̏̑͐̓ ̶̨̢̧͙̲͇͙̬̭̝͍̪̀͋̂͂̈́̌̅̍̈́͘m̵̧͙̟̭͎̺͔̗̻̖̳̼̀̒̓̅͛̎̓̈́̓̈̌̉̽̂͜͝ͅi̷͙͉̻̇̆ĺ̶̻͚͓̣̗̫̝͔̠̯̭̩ͅļ̸̛̙̮̣̭͉͆̌͛̐̔̿̊́̔́̌̚͝ͅ.̸̹̮̩̳̼̖̝̬͔͍̳̣̌̀̓̋̑̚͝
