import sqlite3

db_file = '/path/bt/database/wine.db'


def insert(table_name, id, stuff_name):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("INSERT INTO " + table_name + " (id, name) VALUES (?,?)", (id,stuff_name,))
        conn.commit()
        conn.close()
    except Exception:
        conn.commit()
        conn.close()


def select_by_name(table_name, stuff_name):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        data = c.execute("SELECT id FROM " + table_name + " WHERE name = ?", (stuff_name,))
        data = [x for x in data]
        count = len(data)
        conn.commit()
        conn.close()
        return data
    except Exception:
        conn.commit()
        conn.close()


def select_names(table_name):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        data = c.execute("SELECT name FROM " + table_name)
        data = [x for x in data]
        conn.commit()
        conn.close()
        return data
    except Exception:
        conn.commit()
        conn.close()


def select_ids(table_name):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        data = c.execute("SELECT id FROM " + table_name)
        data = [x for x in data]
        conn.commit()
        conn.close()
        return data
    except Exception:
        conn.commit()
        conn.close()


if __name__ == '__main__':
    a = select_by_name('country', 'Brazil')
    print(a[0][0])
