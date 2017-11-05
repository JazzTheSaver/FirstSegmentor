
include_directories(
  model
)
add_executable(SemiEmbLSTMSegmentor SemiEmbLSTMSegmentor.cpp)
target_link_libraries(SemiEmbLSTMSegmentor ${LIBS})
