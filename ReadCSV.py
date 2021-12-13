import csv
from HashTable import HashMap

# Read CSV file for packages
with open('./data/video_game_sales.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    hash_map = HashMap()

    # Populate Hash Table -- O(n)
    for row in readCSV:
        id = row[0]
        name = row[1]
        platform = row[2]
        year = row[3]
        genre = row[4]
        publisher = row[5]
        na_sales = row[6]
        eu_sales = row[7]
        jp_sales = row[8]
        global_sales = row[9]

        game = [id, name, platform, year, genre, publisher, na_sales, eu_sales, jp_sales, global_sales]

        hash_map.insert(id, game)


    # Access all games
    def get_all_games():
        return hash_map
