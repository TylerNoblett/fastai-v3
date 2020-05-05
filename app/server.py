import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
export_file_url_vision = 'https://www.dropbox.com/s/8dtnca7engl8ck4/baylor.pkl?dl=1'
export_file_name_vision = 'export.pkl'
# TODO: move dropbox links to other dropbox service
export_file_url_lyrics = 'https://www.dropbox.com/s/4rxcievdvkiv8e1/lyrics.pkl?dl=1'
export_file_name_lyrics = 'lyrics.pkl'
export_file_url_music = 'https://www.dropbox.com/s/7qyc0tjbifjd0qk/music.pkl?dl=1'
export_file_name_music = 'music.pkl'

vision_learner = None
lyrics_learner = None
music_learner = None

classes = ['pen', 'chair', 'computer', 'desk', 'person', 'car', 'tree']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['https://tylernoblett.github.io'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner(export_file_url, export_file_name, learner_name):
    await download_file(export_file_url, path / export_file_name)
    try:
        learner_name = load_learner(path, export_file_name)
        return learner_name
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()

tasks = [
    asyncio.ensure_future(setup_learner(export_file_url_vision, export_file_name_vision, vision_learner)),
    asyncio.ensure_future(setup_learner(export_file_url_lyrics, export_file_name_lyrics, lyrics_learner)),
    asyncio.ensure_future(setup_learner(export_file_url_music, export_file_name_music, music_learner)),
]
# TODO: remove this [0]?
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = vision_learner.predict(img)[0]
    prediction = str(prediction).capitalize()
    items = {
        'Tree': 'Trees are great!', 
        'Person': 'If you are reading this, you too are a person', 
        'Car': '4 wheels is all you need',
        'Computer': 'A very useful tool',
        'Pen': 'Useful for writing something down!',
        'Desk': 'The perfect place to put your things',
        'Chair': 'Ideal for sitting'
    }

    return JSONResponse({
        'result': prediction,
        'content': items[prediction]
    })

# TODO: add functions and endpoints for beatles
@app.route('/lyrics')
async def return_lyrics(request):
    chorus_lyrics = "".join(lyric_learner.predict("love", 20, temperature=0.75))
    return JSONResponse({
        'lyrics': chorus_lyrics,
    })

@app.route('/robots.txt')
async def get_yaml(request):
    dirname = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
    print('DIRNAME', dirname)
    return FileResponse(f"{dirname}/static/robots.txt")

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
