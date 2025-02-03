import 'package:http/http.dart' as http;
import 'package:bhoomi/helpers/api_helper.dart';
import 'package:bhoomi/helpers/openai_helper.dart';

class CombinedService {
  static Future<void> fetchAndGenerateInsights(double lat, double lon, String type) async {
    try {
      // Fetch data from Google Places API
      List<dynamic> places = await ApiHelper.fetchNearbyData(lat, lon, type);

      // Prepare a prompt for OpenAI based on fetched data
      String prompt = "Here are nearby $type services:\n";
      for (var place in places) {
        prompt += "Place Name: ${place['name']}, Address: ${place['vicinity']}\n";
      }

      prompt += "Can you provide suggestions or insights based on this data?";

      // Fetch generated response from OpenAI GPT
      String aiGeneratedResponse = await OpenAIHelper.getGeneratedResponse(prompt);

      // Display combined result (You can update UI or do something else)
      print('Google Places Data:\n$prompt');
      print('\nOpenAI GPT Generated Insights:\n$aiGeneratedResponse');
    } catch (error) {
      print('Error occurred: $error');
    }
  }
}
