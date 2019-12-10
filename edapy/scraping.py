import csv

# Instagram Scraping Packages
from igramscraper.instagram import Instagram

# Utilities, free proxies website https://openproxy.space/list/
from itertools import cycle
from .utils import batch
from tqdm import tqdm_notebook as tqdm

def get_account_metadata(username):
    instagram = Instagram()
    account = instagram.get_account(username)
    metadata = {}
    metadata['identifier'] = account.identifier
    metadata['username'] = account.username
    metadata['full_name'] = account.full_name
    metadata['biography'] = account.biography
    metadata['external_url'] = account.external_url
    metadata['media_count'] = account.media_count
    metadata['count_followers'] = account.followed_by_count
    metadata['count_follows'] = account.follows_count
    metadata['is_private'] = account.is_private
    metadata['is_verified'] = account.is_verified
    
    return metadata

def get_medias(metadata, filename, n=None):
    instagram = Instagram()
    if n is not None:
        medias = instagram.get_medias(metadata['username'], count=n)
    else:
        medias = instagram.get_medias(metadata['username'], count=metadata['media_count'])

    lists = []
    for media in tqdm(medias):
        lists.append([
            media.identifier,
            media.short_code,
            media.created_time,
            media.caption,
            media.comments_count if hasattr(media, 'commentsCount') else 0,
            media.likes_count,
            media.link,
            media.image_high_resolution_url,
            media.type
        ])

    cols = ['identifier', 'short_code', 'created_time', 'caption', 'comments_count', 'likes_count', 'link', 'image_high_resolution_url', 'type']
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(lists)

    return medias

def get_all_media_comments(medias, filename, proxies):
    # Initialize files
    with open(filename, 'w', encoding="utf-8") as f:
        f.write("media_id,owner_id,owner_username,owner_comment\n")
    
    # Rotating Proxies
    proxy_cycle = cycle(proxies)    
    proxy_str = next(proxy_cycle)
    proxy = {}
    proxy['http'] = 'http://'+proxy_str
    proxy['https'] = 'https://'+proxy_str
    instagram = Instagram()
    instagram.set_proxies(proxy)
    
    # Create Batches
    batches = []
    for x in batch(medias, 10):
        batches.append(x)
        
    # Do the scraping.....
    for b in tqdm(batches):
        for media in b:
            # Try maximum 10 proxies, otherwise skip
            for _ in range(10):
                try:
                    comments = instagram.get_media_comments_by_id(media.identifier, 10000)
                    for comment in comments['comments']:
                        comment_text = comment.text.strip('\n')
                        comment_text = comment_text.replace('\r', '')
                        comment_text = comment_text.replace('\n', ' ')
                        with open(filename, 'a', encoding="utf-8") as f:
                            f.write('{},{},{},"{}"\n'.format(media.identifier, comment.owner.identifier, comment.owner.username, comment_text))
                    break
                except Exception as e:
                    print(e)        
                    proxy_str = next(proxy_cycle)
                    proxy = {}
                    proxy['http'] = 'http://'+proxy_str
                    proxy['https'] = 'https://'+proxy_str
                    instagram = Instagram()
                    instagram.set_proxies(proxy)

