import csv
from HashTable import HashMap

# Read CSV file for packages
with open('./data/video_game_sales.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    hash_map = HashMap()

    game_sales_by_genre = {}
    game_sales_by_platform = {}

    #Skip the header row
    next(readCSV)

    # Populate Hash Table -- O(n)
    for row in readCSV:
        game_id = row[0]
        name = row[1]
        platform = row[2]
        year = row[3]
        genre = row[4]
        publisher = row[5]
        na_sales = row[6]
        eu_sales = row[7]
        jp_sales = row[8]
        other_sales = row[9]
        global_sales = row[10]

        game = [game_id, name, platform, year, genre, publisher, na_sales, eu_sales, jp_sales, other_sales, global_sales]

        # Build dict for sales by genre
        if genre in game_sales_by_genre:
            game_sales_by_genre[genre] = round(game_sales_by_genre[genre] + float(global_sales), 2)
        else:
            game_sales_by_genre[genre] = round(float(global_sales), 2)

        # Build dict for sales by platform
        if platform in game_sales_by_platform:
            game_sales_by_platform[platform] = round(game_sales_by_platform[platform] + float(global_sales), 2)
        else:
            game_sales_by_platform[platform] = round(float(global_sales), 2)

        hash_map.insert(game_id, game)

    # Access all games
    def get_all_games():
        return hash_map

    # Access Sales By Genre
    def get_game_sales_by_genre():
        return game_sales_by_genre

    def get_game_sales_by_platform():
        return game_sales_by_platform
