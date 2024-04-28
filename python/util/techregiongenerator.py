

from typing import AnyStr, TypeVar, List
from dataclasses import dataclass

Item = TypeVar("Item")

@dataclass
class TechRegion:
    city: AnyStr
    num_companies: float
    longitude: float
    latitude: float

    @staticmethod
    def header() -> List[AnyStr]:
        return ['city', 'num_companies', 'longitude', 'latitude']


class TechRegionGenerator(object):
    """ Generate the input to PyGWalker table with geo-location data"""
    def __init__(self, cities: List[AnyStr], num_companies: List[int], filename: AnyStr):
        from geopy.geocoders import Nominatim

        self.filename = filename
        self.cities = cities
        self.num_companies_lst = num_companies
        self.loc = Nominatim(user_agent='Geopy Library')

    def __call__(self) -> bool:
        import csv
        import logging
        # Step 1: Generate the records of type TechRegion
        tech_regions = [
            TechRegion(city, num_companies, self.loc.geocode(city).longitude, self.loc.geocode(city).latitude)
            for index, (city, num_companies)
            in enumerate(zip(self.cities, self.num_companies_lst))
        ]
        # Step 2: Convert to list into a dictionary
        records = [vars(tech_region) for tech_region in tech_regions]
        # Step 3: Store the dictionary in CSV or JSON format, give the file name
        try:
            match self.filename[-4:]:
                case '.csv':
                    with open(self.filename, 'w') as f:
                        writer = csv.DictWriter(f, fieldnames=TechRegion.header())
                        writer.writeheader()
                        for record in records:
                            writer.writerow(record)
                    return True

                case 'json':
                    import json
                    json_repr = json.dumps(records, indent=4)
                    with open(self.filename, 'w') as f:
                        f.write(json_repr)
                    return True

                case _:
                    logging.error(f'Extension for {self.filename} is incorrect')
                    return False
        except Exception as e:
            logging.error(f'Failed to store object {str(e)}')
            return True


if __name__ == '__main__':
    from collections import UserList

    class MyList(UserList):
        def __init__(self, lst):
            super(MyList, self).__init__(lst)
        def pop(self, s: int = None):
            if s is not None:
                super(MyList, self).pop(s)

        def remove(self, s):
            raise RuntimeError("Remove is not supported")

    my_list = MyList([3, 4, 19, 8])
    my_list.append(88)
    print(my_list.data)
    my_list.pop(2)
    print(my_list.data)
    my_list.remove(8)

    """
    cities = ['San Francisco', 'Los Angeles', 'Santa Barbara', 'San Diego', 'Sacramento', 'San Jose', 'Newport Beach']
    num_companies_list = [2314, 910, 290, 583, 419, 1893, 380]

    my_filename = '../input/locations.csv'
    rand_data_generator = TechRegionGenerator(cities, num_companies_list, my_filename)
    rand_data_generator()
    """
