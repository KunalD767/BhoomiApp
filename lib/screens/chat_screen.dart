import 'package:flutter/material.dart';

class ChatScreen extends StatelessWidget {
  const ChatScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chat with Bhoomi'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text('How can I help you?'),
            ElevatedButton(
              onPressed: () {
               
              },
              child: const Text('Send Message'),
            ),
          ],
        ),
      ),
    );
  }
}
