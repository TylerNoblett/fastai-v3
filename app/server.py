import aiohttp
import asyncio
import uvicorn
import random
import shlex
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

# TODO: remove learner name in params and rename below
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
# TODO: remove learner name in args
tasks = [
    asyncio.ensure_future(setup_learner(export_file_url_vision, export_file_name_vision, vision_learner)),
    asyncio.ensure_future(setup_learner(export_file_url_lyrics, export_file_name_lyrics, lyrics_learner)),
    asyncio.ensure_future(setup_learner(export_file_url_music, export_file_name_music, music_learner)),
]
vision_learner, lyrics_learner, music_learner = loop.run_until_complete(asyncio.gather(*tasks))
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
@app.route('/new-song')
async def return_song(request):
    def create_song(request):
        bridge_words = ["Love", "Can\'t", "Day", "Eight", "ah", "When", "Help", "Nah", "I\'m", "That", "And", "She", "Because", "Yeah", "you\'re", "I", "Life"]
        verse_words = ["There\'s", "Something", "Nothing", "Can\'t", "Ooh", "Saw", "Here", "He", "Got", "She", "Tried", "Love", "Eleanor", "Father", "Jo", "Sweet", "It\'s", "you", "you", "I", "When", "Now", "Hey", "baby\'s", "baby", "Oh", "Yeah", "Lady", "When", "And", "For", "I", "Love", "Dear", "It\'s", "If", "In", "On", "Behind", "You", "I", "She", "Try", "Think", "In", "And", "So", "yesterday", "As", "Suddenly", "yesterday"]
        chorus_words = ["all", "We", "Why", "Penny", "We", "She", "I", "She\'s", "Don\'t", "We","Paperback", "She", "Let", "Someone", "Can\'t", "hold", "Come", "All", "She", "Get", "Oh", "When", "Hello", "You", "I", "Help", "And", "So", "I\'m", "She\'s", "I", "Tuesday", "See", "Friday"]
    #     bridge_words = [x.lower() for x in bridge_words]       
    #     verse_words = [x.lower() for x in verse_words]
    #     chorus_words = [x.lower() for x in chorus_words]

        # grab random word from possible vals
        verse_start = random.choice(verse_words)
        verse2_start = random.choice(verse_words)
        chorus_start = random.choice(chorus_words)
        bridge_start = random.choice(bridge_words)

        # create lyrics
        def create_lyrics(song_section):
            return "".join(lyrics_learner.predict(song_section, 50, temperature=0.75))

        chorus_lyrics = create_lyrics(chorus_start)
        verse_lyrics = create_lyrics(verse_start)
        verse2_lyrics = create_lyrics(verse2_start)
        bridge_lyrics = create_lyrics(bridge_start)

        def remove_stray_quote_marks(string):
            new_string = string.replace("'", '')
            return new_string.replace('"', '')

        # remove quote marks that don't have a partner
        cleaned_chorus_lyrics = remove_stray_quote_marks(chorus_lyrics)
        cleaned_verse_lyrics = remove_stray_quote_marks(verse_lyrics)
        cleaned_verse2_lyrics = remove_stray_quote_marks(verse2_lyrics)
        cleaned_bridge_lyrics = remove_stray_quote_marks(bridge_lyrics)

        # split the lyric string into an array
        chorus_lyrics_split = shlex.split(cleaned_chorus_lyrics)
        verse_lyrics_split = shlex.split(cleaned_verse_lyrics)
        verse2_lyrics_split = shlex.split(cleaned_verse2_lyrics)
        bridge_lyrics_split = shlex.split(cleaned_bridge_lyrics)

        def pick_key():
            keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            key_weights = [4 / 22, 1 / 22, 3 / 22, 2 / 22, 1 / 22, 2 / 22, 9 / 22 ]
            key = random.choices(
                     population=keys,
                     weights=key_weights,
                     k=1)
            return key[0]    

        def pick_tempo():
            tempos = [138, 98, 120, 102, 125, 65, 72, 112, 150, 165, 108, 138, 70, 78, 94, 99, 123, 138, 138, 137, 171, 100]
            return random.choice(tempos)    

        # create the response object
        response_obj = {
            'tempo': pick_tempo(),
            'key': pick_key(),
            'verse_one': [],
            'verse_two': [],
            'verse_three': [],
            'chorus': [],
            'bridge': []
        }

        # takes a song section and makes it into a list (if it's not)
        def clean_line(line):
            start = line.find('[')
            clean_start = line[start:]
            clean_end = ']'.join(clean_start.split(']')[:-1]) + ']'
            return clean_end

        # takes a chord and makes it more human readable
        def make_chords_readable(chord):
            try:
                lower_chord = chord.lower().strip()
            except AttributeError:
                lower_chord = 'imaj'
            chord_list = {
                'isus': 'Isus',
                'i6': 'I6',
                'imaj': 'I',
                'imaj7': 'I7',
                'imaj / viidim': 'I/vii',
                'imin': 'i',
                'iimaj': 'II',
                'iimaj7': 'II7',
                'iimin': 'ii',
                'iimin7': 'ii7',
                'biiimaj': 'bIII',
                'iiimin': 'iii',
                'iiimin7': 'iii7',
                'iiimaj': 'III',
                'iiimaj7': 'III7',
                'ivmaj': 'IV',
                'ivmaj7': 'IV7',
                'ivmaj / vmaj': 'IV/V',
                'ivmin': 'iv',
                'vmaj': 'V',
                'vmaj7': 'V7',
                'bvimaj': 'bVI',
                'vimin': 'vi',
                'vimin7': 'vi7',
                'vimin / vmaj': 'vi/V',
                'vimaj': 'VI',
                'vimaj7': 'VI7',
                'bviimaj': 'VII',
                'xxbridge': 'V',
                'xxchorus': 'IV',
                'xxverse': 'I',
                'w': 'IV',
                'n': 'V',
                'c': 'iv'
            }
            try:
                return chord_list[lower_chord]
            except KeyError:
                return 'I'

        # TODO: what does this do?
        def simplify_chords(piece_str):
            piece_list = json.loads(piece_str)
            mod_piece_list = []
            for i, piece in enumerate(piece_list):
                chord = make_chords_readable(piece[0])
                new_piece = []
                new_piece.append(chord)
                new_piece.extend([piece[1], piece[2], piece[3]])
                mod_piece_list.append(new_piece)
            return mod_piece_list

        # Create 'music'
        verse_start = 'xxverse'
        chorus_start = 'xxchorus'
        bridge_start = 'xxbridge'
        verse_chars = 500
        chorus_chars = 500
        bridge_chars = 500

        def create_music(section_start, section_characters):
            return clean_line("".join(music_learner.predict(section_start, section_characters, temperature=0.75)))

        verse_str = create_music(verse_start, verse_chars)
        chorus_str = create_music(chorus_start, chorus_chars)
        bridge_str = create_music(bridge_start, bridge_chars)
    
    # TODO: replace this with a function
        verse = simplify_chords('['+ verse_str + ']')
        chorus = simplify_chords('['+ chorus_str + ']')
        bridge = simplify_chords('['+ bridge_str + ']')

        def replace_w_with_word(part_of_song, part_of_song_lyrics):
            new_part_of_song = part_of_song
            new_part_of_song_lyrics = part_of_song_lyrics
            for i, grouping in enumerate(new_part_of_song):
                if grouping[1] == ' w ':
                    grouping[1] = new_part_of_song_lyrics[i]
            return new_part_of_song

        response_obj['verse_one'] = replace_w_with_word(verse, verse_lyrics_split)
        response_obj['verse_two'] = replace_w_with_word(verse, verse2_lyrics_split)
        response_obj['chorus'] = replace_w_with_word(chorus, chorus_lyrics_split)
        response_obj['bridge'] = replace_w_with_word(bridge, bridge_lyrics_split)

        return JSONResponse(response_obj)
    
    def error_handler(request):
        try:
            return create_song(request)
        except json.decoder.JSONDecodeError:
            return error_handler(request)

    return error_handler(request)
        
@app.route('/robots.txt')
async def get_yaml(request):
    dirname = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
    print('DIRNAME', dirname)
    return FileResponse(f"{dirname}/static/robots.txt")

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
