import 'package:flutter/material.dart';

class WarehouseDetailsScreen extends StatelessWidget {
  final Map<String, dynamic> warehouse;

  WarehouseDetailsScreen({required this.warehouse});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(warehouse['name']),
        backgroundColor: Colors.teal[100],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Location: ${warehouse['location']}'),
            Text('Phone: ${warehouse['phone']}'),
            Text('Pricing: ${warehouse['price']}'),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
              },
              child: const Text('Contact'),
            ),
          ],
        ),
      ),
    );
  }
}
