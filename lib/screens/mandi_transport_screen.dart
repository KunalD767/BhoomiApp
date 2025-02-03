import 'package:flutter/material.dart';

class MandisTransportScreen extends StatefulWidget {
  @override
  _MandisTransportScreenState createState() => _MandisTransportScreenState();
}

class _MandisTransportScreenState extends State<MandisTransportScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Mandis and Transport'),
        backgroundColor: Colors.teal[100],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: ListView(
                children: [
                  _buildCategoryCard(context, 'Nearby Warehouses', 'assets/images/warehouse_icon.png', '/warehouse-list'),
                  _buildCategoryCard(context, 'Nearby Mandis', 'assets/images/mandi_icon.png', '/mandi-list'),
                  _buildCategoryCard(context, 'Nearby Transport', 'assets/images/transport_icon.png', '/transport-list'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCategoryCard(BuildContext context, String title, String iconPath, String route) {
    return Card(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(10.0),
      ),
      color: Colors.teal[300],
      child: ListTile(
        leading: Image.asset(iconPath, width: 50),
        title: Text(
          title,
          style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
        ),
        onTap: () {
          Navigator.pushNamed(context, route);
        },
      ),
    );
  }
}
