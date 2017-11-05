
include_directories(
  model
)
add_executable(BiLSTMSegmentor BiLSTMSegmentor.cpp)
target_link_libraries(BiLSTMSegmentor ${LIBS})
