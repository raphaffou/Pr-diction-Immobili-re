#!/usr/bin/env python3
import json
import csv
import sys
import time
import random
import urllib.parse
import urllib.request

BASE_URL = "https://www.bienici.com/realEstateAds.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

SLEEP_SECONDS = 0.3

TOTAL_FROM_RANGE = 10000
SAMPLES = 500


def build_url(offset):
    filters = {
        "size": 1,
        "showAllModels": False,
        "filterType": "rent",
        "propertyType": ["flat"],
        "sortBy": "relevance",
        "sortOrder": "desc",
        "onTheMarket": [True],
        "from": offset
    }

    params = {
        "filters": json.dumps(filters),
        "extensionType": "extendedIfNoResult",
        "enableGoogleStructuredDataAggregates": "true",
        "leadingCount": "1"
    }

    return BASE_URL + "?" + urllib.parse.urlencode(params)


def fetch(offset):
    url = build_url(offset)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract(ad):
    blur = ad.get("blurInfo", {})
    pos = blur.get("position", {})
    district = ad.get("district", {})
    status = ad.get("status", {})

    def clean_text(x):
        if not x:
            return ""
        return " ".join(x.replace("\n", " ").split())

    return {
        # Identifiers
        "id": ad.get("id"),
        "reference": ad.get("reference"),
        "account_type": ad.get("accountType"),
        "account_name": ad.get("accountDisplayName"),
        "customer_id": ad.get("customerId"),

        # Location
        "city": ad.get("city"),
        "postal_code": ad.get("postalCode"),
        "department_code": ad.get("departmentCode"),
        "district_name": district.get("name"),
        "district_label": district.get("libelle"),
        "lat": pos.get("lat"),
        "lon": pos.get("lon"),

        # Price & costs
        "price": ad.get("price"),
        "charges": ad.get("charges"),
        "price_per_m2": ad.get("pricePerSquareMeter"),
        "deposit": ad.get("safetyDeposit"),
        "agency_fee": ad.get("agencyRentalFee"),
        "inventory_fee": ad.get("inventoryOfFixturesFees"),

        # Surface & layout
        "surface": ad.get("surfaceArea"),
        "rooms": ad.get("roomsQuantity"),
        "bedrooms": ad.get("bedroomsQuantity"),
        "bathrooms": ad.get("bathroomsQuantity"),
        "showers": ad.get("showerRoomsQuantity"),
        "floor": ad.get("floor"),
        "floors_total": ad.get("floorQuantity"),

        # Amenities
        "balconies": ad.get("balconyQuantity"),
        "terraces": ad.get("terracesQuantity"),
        "parking_places": ad.get("parkingPlacesQuantity"),
        "cellars": ad.get("cellarsOrUndergroundsQuantity"),
        "elevator": ad.get("hasElevator"),
        "disabled_access": ad.get("isDisabledPeopleFriendly"),
        "furnished": ad.get("isFurnished"),
        "duplex": ad.get("isDuplex"),

        # Energy
        "energy_class": ad.get("energyClassification"),
        "energy_value": ad.get("energyValue"),
        "ghg_class": ad.get("greenhouseGazClassification"),
        "ghg_value": ad.get("greenhouseGazValue"),
        "energy_min_cost": ad.get("minEnergyConsumption"),
        "energy_max_cost": ad.get("maxEnergyConsumption"),
        "energy_diag_date": ad.get("energyPerformanceDiagnosticDate"),

        # Heating
        "heating_type": ad.get("heating"),

        # Dates
        "publication_date": ad.get("publicationDate"),
        "modification_date": ad.get("modificationDate"),

        # Flags
        "new_property": ad.get("newProperty"),
        "exclusive": ad.get("isBienIciExclusive"),
        "ad_created_by_pro": ad.get("adCreatedByPro"),
        "on_market": status.get("onTheMarket"),
        "leading_ad": status.get("isLeading"),

        # Media counts
        "photos_count": len(ad.get("photos", [])),
        "virtual_tours_count": len(ad.get("virtualTours", [])),

        # Text
        "title": clean_text(ad.get("title")),
        "description": clean_text(ad.get("description")),

        # Misc
        "exposition": ad.get("exposition"),
        "year_of_construction": ad.get("yearOfConstruction"),
        "transaction_type": ad.get("transactionType"),
    }


def main():
    writer = None

    for offset in range(0, 30000):
        try:
            data = fetch(offset)
            ads = data.get("realEstateAds", [])
            if not ads:
                continue

            row = extract(ads[0])

            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=row.keys())
                #writer.writeheader()

            writer.writerow(row)
            sys.stdout.flush()
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            pass
            #print(f"# error at offset {offset}: {e}", file=sys.stderr)



if __name__ == "__main__":
    main()

