from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import traceback
from pathlib import Path

# Import your existing classes
from app import UFCPredictionPipeline, FighterMatcher

app = Flask(__name__)
CORS(app)

# Global pipeline instance
pipeline = None
fighter_matcher = None
available_fighters = []


def initialize_pipeline():
    """Initialize the pipeline and load basic data"""
    global pipeline, fighter_matcher, available_fighters
    try:
        pipeline = UFCPredictionPipeline()
        print("Loading data...")
        pipeline.load_data()
        pipeline.merge_data()

        # Create fighter matcher for autocomplete
        fighter_matcher = FighterMatcher(pipeline.merged_data)
        available_fighters = fighter_matcher.unique_fighters

        print(f"Initialized with {len(available_fighters)} fighters")
        return True
    except Exception as e:
        print(f"Failed to initialize pipeline: {str(e)}")
        traceback.print_exc()
        return False


@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")


@app.route("/api/fighters")
def get_fighters():
    """Get list of all available fighters for autocomplete"""
    try:
        return jsonify({
            "success": True,
            "fighters": available_fighters,
            "count": int(len(available_fighters))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/search_fighters")
def search_fighters():
    """Search for fighters with fuzzy matching"""
    query = request.args.get("query", "").strip()

    if not query or len(query) < 2:
        return jsonify({"success": True, "matches": []})

    try:
        matches = fighter_matcher._find_best_matches(query, threshold=0.3, top_k=10)
        results = [
            {"name": str(name), "similarity": float(similarity)}
            for name, similarity in matches
        ]

        return jsonify({"success": True, "matches": results, "query": query})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/validate_fighters", methods=["POST"])
def validate_fighters():
    """Validate that both fighters exist and can be matched"""
    try:
        data = request.get_json()
        red_fighter = data.get("red_fighter", "").strip()
        blue_fighter = data.get("blue_fighter", "").strip()

        if not red_fighter or not blue_fighter:
            return jsonify({"success": False, "error": "Both fighters are required"}), 400

        red_matches = fighter_matcher._find_best_matches(red_fighter, threshold=0.6, top_k=1)
        blue_matches = fighter_matcher._find_best_matches(blue_fighter, threshold=0.6, top_k=1)

        red_matched = red_matches[0][0] if red_matches else None
        blue_matched = blue_matches[0][0] if blue_matches else None

        if not red_matched or not blue_matched:
            return jsonify({
                "success": False,
                "error": "One or both fighters could not be found",
                "red_matched": red_matched,
                "blue_matched": blue_matched,
            }), 400

        filtered_data = pipeline.merged_data[
            (pipeline.merged_data["fighter_red"].isin([red_matched, blue_matched]))
            | (pipeline.merged_data["fighter_blue"].isin([red_matched, blue_matched]))
        ]

        if len(filtered_data) == 0:
            return jsonify({"success": False, "error": "No fight data found"}), 400

        return jsonify({
            "success": True,
            "red_matched": red_matched,
            "blue_matched": blue_matched,
            "data_count": int(len(filtered_data))
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_fight():
    """Make fight prediction"""
    try:
        data = request.get_json()
        red_fighter = data.get("red_fighter", "").strip()
        blue_fighter = data.get("blue_fighter", "").strip()

        if not red_fighter or not blue_fighter:
            return jsonify({"success": False, "error": "Both fighters are required"}), 400

        red_matches = fighter_matcher._find_best_matches(red_fighter, threshold=0.6, top_k=1)
        blue_matches = fighter_matcher._find_best_matches(blue_fighter, threshold=0.6, top_k=1)

        if not red_matches or not blue_matches:
            return jsonify({"success": False, "error": "Could not match fighter names"}), 400

        red_matched = red_matches[0][0]
        blue_matched = blue_matches[0][0]

        # Filter data
        fighter_matcher.matched_names = [red_matched, blue_matched]
        filtered_data = pipeline.merged_data[
            (pipeline.merged_data["fighter_red"].isin([red_matched, blue_matched]))
            | (pipeline.merged_data["fighter_blue"].isin([red_matched, blue_matched]))
        ]

        if len(filtered_data) == 0:
            return jsonify({"success": False, "error": "No fight data found"}), 400

        # Add prediction row
        filtered_data = pipeline._add_prediction_row(filtered_data, [red_matched, blue_matched])

        # Feature engineering
        processed_data = pipeline.engineer_features(filtered_data)
        features = pipeline.prepare_for_prediction(processed_data)

        # Models (including PyTorch)
        model_paths = [
            #("models/adaboostclassifier.pkl", "auto"),
            #("models/decisiontreeclassifier.pkl", "auto"),
            #("models/gaussiannb.pkl", "auto"),
            ("models/gradientboostingclassifier.pkl", "auto"),
            #("models/kneighborsclassifier.pkl", "auto"),
            ("models/logisticregression.pkl", "auto"),
            ("models/PyTorch_state_dict.pth", "Neural Network"),  # Added PyTorch model
            ("models/randomforestclassifier.pkl", "auto"),
            ("models/sgdclassifier.pkl", "auto"),
            ("models/svc.pkl", "auto"),
            #("models/xgbclassifier.pkl", "auto"),
        ]

        results = []
        for model_path, model_type in model_paths:
            try:
                if Path(model_path).exists():
                    result = pipeline.predict_single_model(features, model_path, model_type)

                    # Convert NumPy → Python types for all numeric values
                    model_result = {
                        "model": Path(model_path).stem,
                        "fighter_red_win_prob": float(result["fighter_red_win_prob"]),
                        "fighter_blue_win_prob": float(result["fighter_blue_win_prob"]),
                        "confidence": float(result["confidence"]),
                        "prediction": int(result["prediction"])
                    }
                    results.append(model_result)
                    
                    print(f"✓ {model_result['model']}: Red={model_result['fighter_red_win_prob']:.3f}, Blue={model_result['fighter_blue_win_prob']:.3f}")
                    
                else:
                    print(f"⚠ Model file not found: {model_path}")
                    
            except Exception as model_error:
                print(f"✗ Error with {model_path}: {model_error}")
                # Continue with other models instead of failing completely

        if not results:
            return jsonify({"success": False, "error": "No models made successful predictions"}), 500

        # Ensemble calculation
        avg_red_prob = float(sum(r["fighter_red_win_prob"] for r in results) / len(results))
        avg_blue_prob = 1.0 - avg_red_prob
        ensemble_prediction = "red" if avg_red_prob > 0.5 else "blue"
        winner_name = red_matched if ensemble_prediction == "red" else blue_matched

        # Enhanced response with more details
        response_data = {
            "success": True,
            "fighters": {"red": red_matched, "blue": blue_matched},
            "prediction": {
                "winner": ensemble_prediction,
                "winner_name": winner_name,
                "red_probability": round(avg_red_prob, 3),
                "blue_probability": round(avg_blue_prob, 3),
                "confidence": round(max(avg_red_prob, avg_blue_prob), 3),
            },
            "model_results": results,
            "models_used": len(results),
            "models_attempted": len(model_paths),
            "fight_data_count": int(len(filtered_data) - 1),
            "ensemble_details": {
                "red_votes": sum(1 for r in results if r["prediction"] == 1),
                "blue_votes": sum(1 for r in results if r["prediction"] == 0),
                "unanimous": len(set(r["prediction"] for r in results)) == 1
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    # Check if models exist
    model_paths = [
        "models/adaboostclassifier.pkl",
        "models/decisiontreeclassifier.pkl", 
        "models/gaussiannb.pkl",
        "models/gradientboostingclassifier.pkl",
        "models/kneighborsclassifier.pkl",
        "models/logisticregression.pkl",
        "models/PyTorch_state_dict.pth",  # Check PyTorch model
        "models/randomforestclassifier.pkl",
        "models/sgdclassifier.pkl",
        "models/svc.pkl",
        "models/xgbclassifier.pkl"
    ]
    
    models_available = sum(1 for path in model_paths if Path(path).exists())
    
    return jsonify({
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "fighters_available": int(len(available_fighters)) if available_fighters else 0,
        "models_available": f"{models_available}/{len(model_paths)}",
        "pytorch_model_exists": Path("models/PyTorch_state_dict.pth").exists()
    })


if __name__ == "__main__":
    print("Initializing UFC Prediction API...")
    if initialize_pipeline():
        print("✓ Pipeline initialized successfully")
        
        # Check PyTorch availability
        try:
            import torch
            import torch.nn as nn
            print("✓ PyTorch available")
        except ImportError:
            print("⚠ PyTorch not available - neural network predictions will fail")
            
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("✗ Failed to initialize pipeline")