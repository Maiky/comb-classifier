import psycopg2
import datetime



class Adapter:

    def __init__(self):
        self.conn = psycopg2.connect(database='testdb', user='mareikeziese')
        self.curs = self.conn.cursor()


    def _generateMultipolygonSQL(self, polygons):
        postgisSQL = "MULTIPOLYGON("
        for multipolygon in polygons.values():
            if(len(multipolygon) > 1 or len(multipolygon[0]) > 2):
                tmpSQL = "("
                for single_polygon in multipolygon:
                    if(len(single_polygon)> 2):
                        tmpSQL +="("
                        for point in single_polygon:
                            tmpSQL += str(point[0,0])+ " "+str(point[0,1])+","
                        tmpSQL += str(single_polygon[0,0,0])+ " "+str(single_polygon[0,0,1])+","
                        tmpSQL = tmpSQL[:-1]
                        tmpSQL +="),"
                tmpSQL = tmpSQL[:-1]
                tmpSQL +="),"
                postgisSQL += tmpSQL
        postgisSQL = postgisSQL[:-1] + ")"
        return postgisSQL




    def insert_comb_layout(self, polygons, camera_id, date, class_id):
        polygon_sql = self._generateMultipolygonSQL(polygons)
        #print(polygon_sql)
        self.curs.execute("INSERT INTO comb_distribution (date, camera_id, class_id, area) VALUES "
                          "(%(date)s, %(camera_id)s, %(class_id)s, %(polygon)s); ",
                          {'date':date, 'camera_id': camera_id, 'class_id': class_id, 'polygon': polygon_sql})
        self.conn.commit()

    def get_area_sizes(self, min_time=None, max_time=None, camera_id = None, class_label = None):
        self.curs.execute("")
        print()