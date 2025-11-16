from Methods import get_json, get_filename, put_json
from Heap import HeapFile
from spimi.gestor_indice_spimi import createInvertedIndex, addInvertedIndex, finalizar_indice_spimi
from spimi.buscador import BuscadorSPIMI
import shutil
import psutil
import os

class File:

    def get_pk(self):
        for field in self.relation:
            if "key" in self.relation[field] and (self.relation[field]["key"] == "primary"):
                return field

    def __init__(self, table: str):
        self.filename = get_filename(table)
        self.relation, self.indexes = get_json(self.filename, 2)
        self.primary_key = self.get_pk()

    def p_print(self, name, record, additional, filename = ""):
         print(name)
         print(record)
         print(additional)
         print(filename)

    def insert(self, params):
        
        mainfilename = self.indexes["primary"]["filename"]
        record = params["record"]
        record["deleted"] = False
        additional = {"key": None, "unique": []}
        records = []

        for index in self.indexes:
            if self.indexes[index]["filename"] == mainfilename and index != "primary":
                additional["key"] = index
                break
        
        for field in self.relation:
            if "key" in self.relation[field] and (self.relation[field]["key"] == "primary" or self.relation[field]["key"] == "unique"):
                    additional["unique"].append(field)
        
        maindex = self.indexes["primary"]["index"]
        mainfilename = self.indexes["primary"]["filename"]

        if (maindex == "heap"):
            InsFile = HeapFile(mainfilename)
            records = InsFile.insert(record,additional)
            
        if len(records) >= 1:

            for index in self.indexes:

                if index == "primary" or self.indexes[index]["filename"]  == mainfilename:
                    continue
                
                filename = self.indexes[index]["filename"]
                additional = {"key": index}

                for record in records:

                    new_record = {}

                    if self.indexes["primary"]["index"] == "heap":
                        new_record = {"pos": record[1], "deleted": False}
                        new_record[index] = record[0][index]
                    
                    else:
                        new_record = {"pk": record[self.primary_key], "deleted": False}
                        new_record[index] = record[index]

                    indx = self.indexes[index]["index"]

                    if (indx  == "hash"):
                        self.p_print("hash", new_record,additional,filename) 
                    elif (indx == "b+"):
                        self.p_print("b+", new_record,additional,filename) 
                    elif (indx == "rtree"):
                        self.p_print("rtree", new_record,additional,filename)
                     

    def create_inverted_text_index(self, column: str):

        if column not in self.relation or column in self.indexes:
            return
        
        if self.relation[column]["type"] not in ["c", "char", "s", "varchar", "string"]:
            return
        
        routes = self.filename.split("/")
        inverted_folder = f"{routes[0]}/{routes[1]}/inverted_text/"
        self.indexes[column] = {"index": "inverted_text", "folder": inverted_folder}
        put_json(self.filename, [self.relation, self.indexes])
        os.makedirs(inverted_folder, exist_ok=True)
        
        mainfilename = self.indexes["primary"]["filename"]
        RecFile = HeapFile(mainfilename)
        mem = psutil.virtual_memory()

        total_records = RecFile.get_number_records()
        records_for_ram = mem.available // RecFile.REC_SIZE

        if (records_for_ram >= total_records):
            inicio = 0
            salto = total_records
            records = RecFile.get_range_records(inicio, inicio+salto)
            createInvertedIndex(f"{inverted_folder}bloques", records, "description", "pos")
            finalizar_indice_spimi(f"{inverted_folder}bloques")
        else:
            inicio = 0
            salto = records_for_ram
            records = RecFile.get_range_records(0, inicio+salto)

            createInvertedIndex(f"{inverted_folder}bloques", records, "description", "pos")
            
            inicio = records_for_ram

            while inicio < total_records:

                records = RecFile.get_range_records(inicio, inicio+records_for_ram)

                addInvertedIndex(f"{inverted_folder}bloques", records, "description", "pos")

                inicio += salto

    def drop_inverted_text_index(self, column: str):

        if column not in self.relation or column not in self.indexes:
            return
        
        exists = False
        folder = ""

        for index in self.indexes:
             if index == column and self.indexes[index]["index"] == "inverted_text":
                exists = True
                folder = self.indexes[index]["folder"]
                break

        if exists:
            del self.indexes[column]
            put_json(self.filename, [self.relation, self.indexes])
            shutil.rmtree(folder)

    def text_search(self, column: str, consulta: str, k: int):
        if column not in self.relation or column not in self.indexes:
            return
        
        exists = False
        folder = ""

        for index in self.indexes:
             if index == column and self.indexes[index]["index"] == "inverted_text":
                exists = True
                folder = self.indexes[index]["folder"]
                break
        
        records = []
    
        if exists:
            buscador = BuscadorSPIMI(f"{folder}bloques")
            ranking = buscador.rankear_consulta(consulta, k)
            mainfilename = self.indexes["primary"]["filename"]
            RecFile = HeapFile(mainfilename)
            records = RecFile.search_using_pos_ranking(ranking)
    
        return records

    def execute(self, params: dict):

        if params["op"] == "insert":
            self.insert(params)
        
        elif params["op"] == "create_inverted_text":
            self.create_inverted_text_index(params["column"])
        
        elif params["op"] == "drop_inverted_text":
            self.drop_inverted_text_index(params["column"])
        
        elif params["op"] == "text_search":
            return self.text_search(params["column"], params["consulta"], params["k"])