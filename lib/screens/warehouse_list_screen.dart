import 'package:flutter/material.dart';
import 'package:bhoomi/helpers/location_helper.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:geolocator/geolocator.dart';
import 'config.dart';

class WarehouseListScreen extends StatefulWidget {
  @override
  _WarehouseListScreenState createState() => _WarehouseListScreenState();
}

class _WarehouseListScreenState extends State<WarehouseListScreen> {
  late Future<List<dynamic>> _nearbyWarehouses; // Initialize as a Future

  @override
  void initState() {
    super.initState();
    _nearbyWarehouses = _fetchNearbyWarehouses(); // Set the Future
  }

  Future<List<dynamic>> _fetchNearbyWarehouses() async {
    try {
      Position position = await LocationHelper.getCurrentLocation();
      final warehouseList = await fetchWarehouseData(
          position.latitude, position.longitude);
      return warehouseList;
    } catch (e) {
      print(e);
      return [];
    }
  }

  Future<List<dynamic>> fetchWarehouseData(double latitude, double longitude) async {
    try {
final url = Uri.parse('$baseUrl/api/warehouses'); // Make sure this matches the backend route
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: json.encode({"latitude": latitude, "longitude": longitude}),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['recommendations'] != null && data['recommendations'] is List) {
          return data['recommendations'];
        } else {
          print("Unexpected response format: recommendations is not a list");
          return [];
        }
      } else {
        throw Exception("Failed to load warehouse data");
      }
    } catch (e) {
      print("Error fetching warehouse data: $e");
      return [];
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Nearby Warehouse Services'),
        backgroundColor: Colors.teal[100],
      ),
      body: FutureBuilder<List<dynamic>>(
        future: _nearbyWarehouses,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return Center(child: Text('No warehouse services found nearby.'));
          } else {
            return ListView.builder(
              itemCount: snapshot.data!.length,
              itemBuilder: (context, index) {
                final warehouse = snapshot.data![index];
                final distance = warehouse['distance']?.toStringAsFixed(2) ?? 'N/A';
                final rating = warehouse['rating']?.toString() ?? 'No rating available';

                return Container(
                  margin: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.black),
                    borderRadius: BorderRadius.circular(10),
                    color: Colors.teal[50],
                  ),
                  child: ListTile(
                    title: Text(warehouse['name']),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Rating: $rating'),
                        Text('Distance: $distance km'),
                      ],
                    ),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: List.generate(
                        warehouse['rating'] != null ? warehouse['rating'].round() : 0,
                        (index) => Icon(Icons.star, color: Colors.yellow),
                      ),
                    ),
                    onTap: () {
                      Navigator.pushNamed(context, '/warehouseDetails', arguments: warehouse);
                    },
                  ),
                );
              },
            );
          }
        },
      ),
    );
  }
}
